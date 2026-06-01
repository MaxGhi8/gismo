/** @file gsNeuralPrec.h

    @brief ONNX-based neural-network preconditioner.

    Wraps an ONNX Runtime session in a gsLinearOperator<T>, so an exported
    PyTorch / Flax / ... model can be used as a preconditioner inside
    gsConjugateGradient (or any other Krylov solver in G+Smo).

    Two classes are provided:

      - gsNeuralModel<T>: owns the heavy ORT state — Env, Session, model
        weights (on GPU when CUDA is used). Construct once, share across
        many gsNeuralPrec instances. ORT sessions are thread-safe for
        concurrent Run() calls.

      - gsNeuralPrec<T>: cheap per-call wrapper holding its own input/output
        buffers and Ort::Value views. Each instance binds its own auxiliary
        inputs (e.g. per-patch geometry features) and exposes apply() as a
        gsLinearOperator. Multiple instances can share a single
        gsNeuralModel.

    The model is loaded once. Per-call buffers, Ort::Value views and the
    name pointer lists are all preallocated in the constructor, so apply()
    does only a float<->real_t cast in, session.Run(), and a cast out.

    Build: this header requires onnxruntime_cxx_api.h on the include path
    and the onnxruntime shared library on the link line. See
    examples/CMakeLists.txt for the ONNXRUNTIME_ROOT integration.

    Author(s): M. Ghiotto
*/

#pragma once

#include <gismo.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace gismo {

/// @brief Shared ONNX Runtime session + cached I/O metadata.
///
/// Loading an ONNX model is expensive: disk I/O, graph optimization,
/// per-EP kernel allocation, and on CUDA the upload of model weights to
/// GPU. Construct one gsNeuralModel per ONNX file and share it among all
/// the gsNeuralPrec instances that need it (e.g. one per IETI patch).
///
/// \tparam T scalar type used by G+Smo (only carried as a tag; ONNX
///         tensors are float32 regardless).
template <class T>
class gsNeuralModel
{
public:
    typedef memory::shared_ptr<gsNeuralModel> Ptr;
    typedef memory::unique_ptr<gsNeuralModel> uPtr;

    gsNeuralModel(const std::string & model_path,
                  bool   use_cuda         = false,
                  int    cuda_device_id   = 0,
                  int    intra_op_threads = 1)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "gsNeuralModel")
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(intra_op_threads);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        if (use_cuda)
        {
            // Use V2 API to disable TF32: on Ampere/Ada GPUs TF32 reduces
            // float32 matmul mantissa to 10 bits, causing O(1e-3) errors vs CPU.
            Ort::CUDAProviderOptions cuda_options;
            cuda_options.Update({{"device_id", std::to_string(cuda_device_id)},
                                 {"use_tf32",  "0"}});
            session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
        }

        m_session.reset(new Ort::Session(m_env, model_path.c_str(), session_options));

        Ort::AllocatorWithDefaultOptions allocator;

        const size_t n_inputs = m_session->GetInputCount();
        m_inputNameStrs.reserve(n_inputs);
        m_inputShapes.resize(n_inputs);

        for (size_t i = 0; i < n_inputs; ++i)
        {
            auto name_ptr = m_session->GetInputNameAllocated(i, allocator);
            m_inputNameStrs.emplace_back(name_ptr.get());

            auto typeinfo_i = m_session->GetInputTypeInfo(i);  // keep alive
            auto tinfo = typeinfo_i.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tinfo.GetShape();
            // Dynamic dims (-1) are replaced with 1 (typical for the batch axis).
            for (auto & d : shape) if (d < 0) d = 1;
            m_inputShapes[i] = shape;
        }

        GISMO_ENSURE(m_session->GetOutputCount() == 1,
            "gsNeuralModel: model must have exactly one output, has " << m_session->GetOutputCount());

        auto out_name_ptr = m_session->GetOutputNameAllocated(0, allocator);
        m_outputNameStr   = out_name_ptr.get();

        auto typeinfo_out = m_session->GetOutputTypeInfo(0);  // keep alive
        auto otinfo = typeinfo_out.GetTensorTypeAndShapeInfo();
        m_outputShape = otinfo.GetShape();
        for (auto & d : m_outputShape) if (d < 0) d = 1;
    }

    /// @brief Underlying ORT session. Mutable: Run() is non-const but the
    /// model state it reads from is logically const after construction.
    Ort::Session & session() const { return *m_session; }

    size_t numInputs() const { return m_inputNameStrs.size(); }
    const std::vector<std::string>          & inputNames()  const { return m_inputNameStrs; }
    const std::vector<std::vector<int64_t>> & inputShapes() const { return m_inputShapes; }
    const std::string                       & outputName()  const { return m_outputNameStr; }
    const std::vector<int64_t>              & outputShape() const { return m_outputShape; }

private:
    Ort::Env                              m_env;
    mutable std::unique_ptr<Ort::Session> m_session;     // Run() is non-const
    std::vector<std::string>              m_inputNameStrs;
    std::vector<std::vector<int64_t>>     m_inputShapes;
    std::string                           m_outputNameStr;
    std::vector<int64_t>                  m_outputShape;
};

/// @brief Neural-network preconditioner backed by a shared gsNeuralModel.
///
/// The model must have N >= 1 named inputs and exactly one named output.
/// One input is designated as the "primary" input at construction time;
/// it receives the residual vector on every apply() call. Every other
/// input must be bound via setAuxiliaryInput() before the first apply().
///
/// Many instances can share one gsNeuralModel — this is the recommended
/// pattern when one model is applied to multiple subdomains (e.g. IETI
/// patches): construct the gsNeuralModel once, then construct one
/// gsNeuralPrec per subdomain and bind that subdomain's auxiliary inputs.
///
/// \tparam T scalar type used by G+Smo (typically real_t). ONNX tensors
///         are float32; casts happen inside apply().
template <class T>
class gsNeuralPrec : public gsLinearOperator<T>
{
public:
    typedef memory::shared_ptr<gsNeuralPrec> Ptr;
    typedef memory::unique_ptr<gsNeuralPrec> uPtr;

    /// @brief Construct from a shared model. Cheap: allocates only the
    /// per-instance I/O float buffers and Ort::Value views.
    gsNeuralPrec(typename gsNeuralModel<T>::Ptr model,
                 const std::string & primary_input_name,
                 const std::string & output_name)
    : m_model(model),
      m_primaryInputName(primary_input_name),
      m_outputName(output_name),
      m_memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      m_outputValue(nullptr)
    {
        const auto & names  = m_model->inputNames();
        const auto & shapes = m_model->inputShapes();
        const size_t n_inputs = m_model->numInputs();

        m_inputNames.reserve(n_inputs);
        m_inputBuffers.resize(n_inputs);
        m_inputShapes = shapes;                    // local mutable copy for Ort::Value
        m_inputValues.reserve(n_inputs);
        m_inputBound.assign(n_inputs, false);

        index_t primaryIdx = -1;
        for (size_t i = 0; i < n_inputs; ++i)
        {
            m_inputNames.push_back(names[i].c_str());

            int64_t numel = 1;
            for (auto d : shapes[i]) numel *= d;
            m_inputBuffers[i].assign(static_cast<size_t>(numel), 0.0f);

            m_inputValues.push_back(Ort::Value::CreateTensor<float>(
                m_memInfo,
                m_inputBuffers[i].data(),
                static_cast<size_t>(numel),
                m_inputShapes[i].data(),
                m_inputShapes[i].size()));

            if (names[i] == m_primaryInputName)
            {
                primaryIdx = static_cast<index_t>(i);
                // Primary slot is "bound" by every apply() call, so mark it now
                // to keep the readiness check simple.
                m_inputBound[i] = true;
            }
        }

        if (primaryIdx < 0)
        {
            std::ostringstream oss;
            oss << "gsNeuralPrec: primary input '" << m_primaryInputName
                << "' not found in model. Available inputs:";
            for (const auto & n : names) oss << " " << n;
            GISMO_ERROR(oss.str());
        }
        m_primaryIdx = primaryIdx;

        GISMO_ENSURE(m_model->outputName() == m_outputName,
            "gsNeuralPrec: requested output '" << m_outputName
            << "' does not match the model's output '" << m_model->outputName() << "'");
        m_outputNamePtr = m_model->outputName().c_str();

        m_outputShape = m_model->outputShape();    // local mutable copy for Ort::Value
        int64_t out_numel = 1;
        for (auto d : m_outputShape) out_numel *= d;
        m_outputBuffer.assign(static_cast<size_t>(out_numel), 0.0f);
        m_outputValue = Ort::Value::CreateTensor<float>(
            m_memInfo,
            m_outputBuffer.data(),
            static_cast<size_t>(out_numel),
            m_outputShape.data(),
            m_outputShape.size());

        m_rows = static_cast<index_t>(m_inputBuffers[m_primaryIdx].size());
        m_cols = static_cast<index_t>(m_outputBuffer.size());
    }

    /// @brief Convenience constructor: load the model and wrap it in one
    /// step. For repeated use across many subdomains, build a shared
    /// gsNeuralModel once and pass it to the other constructor instead.
    gsNeuralPrec(const std::string & model_path,
                 const std::string & primary_input_name,
                 const std::string & output_name,
                 bool   use_cuda         = false,
                 int    cuda_device_id   = 0,
                 int    intra_op_threads = 1)
    : gsNeuralPrec(
        typename gsNeuralModel<T>::Ptr(new gsNeuralModel<T>(
            model_path, use_cuda, cuda_device_id, intra_op_threads)),
        primary_input_name, output_name)
    {}

    /// @brief Bind (or update) a non-primary input. Data is cast to float and
    /// copied into the preallocated input buffer; the bound value is reused
    /// on every subsequent apply() until this method is called again.
    void setAuxiliaryInput(const std::string & name, const gsMatrix<T> & data)
    {
        const index_t idx = findInputIndex(name);
        GISMO_ASSERT(idx != m_primaryIdx,
            "gsNeuralPrec: '" << name << "' is the primary input; "
            "it is filled by apply() and cannot be set manually.");
        GISMO_ASSERT(static_cast<size_t>(data.size()) == m_inputBuffers[idx].size(),
            "gsNeuralPrec: shape mismatch for input '" << name << "': model expects "
            << m_inputBuffers[idx].size() << " values, got " << data.size());

        castIn(data.data(), m_inputBuffers[idx].data(), data.size());
        m_inputBound[idx] = true;
    }

    void apply(const gsMatrix<T> & input, gsMatrix<T> & x) const override
    {
        GISMO_ASSERT(input.size() == m_rows,
            "gsNeuralPrec::apply: input has size " << input.size()
            << ", expected " << m_rows);

        for (size_t i = 0; i < m_inputBound.size(); ++i)
            GISMO_ASSERT(m_inputBound[i],
                "gsNeuralPrec::apply: auxiliary input '" << m_model->inputNames()[i]
                << "' has not been bound. Call setAuxiliaryInput() first.");

        castIn(input.data(), m_inputBuffers[m_primaryIdx].data(), input.size());

        m_model->session().Run(m_runOptions,
                               m_inputNames.data(),
                               m_inputValues.data(),
                               m_inputValues.size(),
                               &m_outputNamePtr,
                               &m_outputValue,
                               1);

        x.resize(m_cols, 1);
        castOut(m_outputBuffer.data(), x.data(), m_cols);
    }

    index_t rows() const override { return m_rows; }
    index_t cols() const override { return m_cols; }

    /// @brief Underlying shared model; useful for building more instances
    /// that share the same ORT session.
    typename gsNeuralModel<T>::Ptr model() const { return m_model; }

private:
    static void castIn(const T * src, float * dst, index_t n)
    {
        for (index_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    }
    static void castOut(const float * src, T * dst, index_t n)
    {
        for (index_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
    }

    index_t findInputIndex(const std::string & name) const
    {
        const auto & names = m_model->inputNames();
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return static_cast<index_t>(i);
        GISMO_ERROR("gsNeuralPrec: input '" << name << "' not found in model.");
    }

    typename gsNeuralModel<T>::Ptr m_model;

    std::string              m_primaryInputName;
    std::string              m_outputName;
    std::vector<const char*> m_inputNames;      // pointers into m_model->inputNames()
    const char *             m_outputNamePtr = nullptr;

    Ort::MemoryInfo                   m_memInfo;
    Ort::RunOptions                   m_runOptions{nullptr};

    // Buffers viewed by m_inputValues / m_outputValue. Mutable because apply()
    // is logically const but must rewrite the input bytes and receive output.
    mutable std::vector<std::vector<float>> m_inputBuffers;
    std::vector<std::vector<int64_t>>       m_inputShapes;   // local copy
    mutable std::vector<Ort::Value>         m_inputValues;
    std::vector<bool>                       m_inputBound;

    mutable std::vector<float> m_outputBuffer;
    std::vector<int64_t>       m_outputShape;                // local copy
    mutable Ort::Value         m_outputValue;

    index_t m_primaryIdx = -1;
    index_t m_rows = 0;
    index_t m_cols = 0;
};

} // namespace gismo
