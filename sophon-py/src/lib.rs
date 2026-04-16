use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use output_compressor::OutputCompressor;
use prompt_compressor::PromptCompressor;
use sophon_core::tokens::count_tokens;

#[pyclass]
struct Sophon {
    prompt: PromptCompressor,
    output: OutputCompressor,
}

#[pymethods]
impl Sophon {
    #[new]
    fn new() -> Self {
        Self {
            prompt: PromptCompressor::default(),
            output: OutputCompressor::default(),
        }
    }

    #[pyo3(signature = (prompt, query, max_tokens=None))]
    fn compress_prompt(
        &mut self,
        prompt: &str,
        query: &str,
        max_tokens: Option<usize>,
    ) -> PyResult<PromptResult> {
        let result = self
            .prompt
            .compress(prompt, query, None, max_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PromptResult {
            compressed_prompt: result.compressed_prompt,
            token_count: result.token_count,
            included_sections: result.included_sections,
            excluded_sections: result.excluded_sections,
            compression_ratio: result.compression_ratio,
        })
    }

    fn compress_output(&self, command: &str, output: &str) -> OutputResult {
        let result = self.output.compress(command, output);
        OutputResult {
            compressed: result.compressed,
            original_tokens: result.original_tokens,
            compressed_tokens: result.compressed_tokens,
            ratio: result.ratio,
            filter_name: result.filter_name,
        }
    }

    #[staticmethod]
    fn count_tokens(text: &str) -> usize {
        count_tokens(text)
    }
}

#[pyclass(get_all)]
#[derive(Clone)]
struct PromptResult {
    compressed_prompt: String,
    token_count: usize,
    included_sections: Vec<String>,
    excluded_sections: Vec<String>,
    compression_ratio: f32,
}

#[pymethods]
impl PromptResult {
    fn __repr__(&self) -> String {
        format!(
            "PromptResult(tokens={}, ratio={:.2}, sections={})",
            self.token_count,
            self.compression_ratio,
            self.included_sections.len()
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone)]
struct OutputResult {
    compressed: String,
    original_tokens: usize,
    compressed_tokens: usize,
    ratio: f32,
    filter_name: String,
}

#[pymethods]
impl OutputResult {
    fn __repr__(&self) -> String {
        format!(
            "OutputResult(orig={}, comp={}, ratio={:.2}, filter={})",
            self.original_tokens, self.compressed_tokens, self.ratio, self.filter_name
        )
    }
}

#[pymodule]
fn _sophon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sophon>()?;
    m.add_class::<PromptResult>()?;
    m.add_class::<OutputResult>()?;
    Ok(())
}
