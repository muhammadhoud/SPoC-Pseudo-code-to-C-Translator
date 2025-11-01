"""
Streamlit App for SPoC Code Generator
Downloads model from GitHub Releases (no Hugging Face needed)
Complete production-ready version
"""

import streamlit as st
import torch
import os
import zipfile
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES WITH YOUR GITHUB INFO
# ============================================================================
GITHUB_REPO = "muhammadhoud/SPoC-Pseudo-code-to-C-Translator"  # ← CHANGE THIS! 
MODEL_RELEASE_TAG = "v1.0.0"                       # Your release tag
MODEL_ZIP_NAME = "final_spoc_model.zip"            # Name of your zip file
MODEL_DIR = "final_spoc_model"                     # Directory name after extraction

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SPoC Code Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    .download-progress {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DOWNLOAD FUNCTIONS
# ============================================================================

def download_file_with_progress(url, filename):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            # No content-length header, download directly
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        chunk_size = 8192
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    downloaded += len(chunk)
                    f.write(chunk)
                    
                    # Update progress
                    progress = downloaded / total_size
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB "
                        f"({progress*100:.1f}%)"
                    )
        
        progress_bar.empty()
        status_text.empty()
        return True
        
    except requests.exceptions.RequestException as e:
        st.error(f"Download failed: {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False

def check_if_chunked():
    """
    Check if model is split into chunks by looking for metadata file
    
    Returns:
        tuple: (is_chunked: bool, metadata: dict or None)
    """
    metadata_url = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/{MODEL_ZIP_NAME}.metadata.json"
    
    try:
        response = requests.get(metadata_url, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def download_and_merge_chunks(metadata):
    """
    Download and merge chunked model files
    
    Args:
        metadata: Dictionary containing chunk information
        
    Returns:
        bool: True if successful, False otherwise
    """
    total_chunks = metadata['total_chunks']
    
    st.info(f"📦 Model is split into {total_chunks} chunks. Downloading all parts...")
    
    try:
        # Download each chunk
        for i in range(1, total_chunks + 1):
            chunk_name = f"{MODEL_ZIP_NAME}.part{i:03d}"
            chunk_url = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/{chunk_name}"
            
            st.write(f"📥 Downloading chunk {i}/{total_chunks}: {chunk_name}")
            
            if not download_file_with_progress(chunk_url, chunk_name):
                st.error(f"Failed to download chunk {i}")
                return False
        
        # Merge chunks
        st.write("🔗 Merging chunks into single file...")
        merge_progress = st.progress(0)
        
        with open(MODEL_ZIP_NAME, 'wb') as merged_file:
            for i in range(1, total_chunks + 1):
                chunk_name = f"{MODEL_ZIP_NAME}.part{i:03d}"
                
                if not os.path.exists(chunk_name):
                    st.error(f"Chunk file missing: {chunk_name}")
                    return False
                
                with open(chunk_name, 'rb') as chunk_file:
                    merged_file.write(chunk_file.read())
                
                # Clean up chunk file
                os.remove(chunk_name)
                
                # Update progress
                merge_progress.progress(i / total_chunks)
        
        merge_progress.empty()
        st.success("✅ Chunks merged successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error during chunk processing: {e}")
        return False

def download_single_file():
    """
    Download single model file (not chunked)
    
    Returns:
        bool: True if successful, False otherwise
    """
    download_url = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_RELEASE_TAG}/{MODEL_ZIP_NAME}"
    
    st.info("📥 Downloading model from GitHub Release...")
    st.caption(f"Source: {download_url}")
    
    return download_file_with_progress(download_url, MODEL_ZIP_NAME)

def extract_model():
    """
    Extract model from zip file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(MODEL_ZIP_NAME):
            st.error(f"Zip file not found: {MODEL_ZIP_NAME}")
            return False
        
        st.write("📂 Extracting model files...")
        
        with zipfile.ZipFile(MODEL_ZIP_NAME, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            if total_files == 0:
                st.error("Zip file is empty!")
                return False
            
            # Extract with progress
            extract_progress = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(file_list):
                zip_ref.extract(file, '.')
                extract_progress.progress((idx + 1) / total_files)
                status_text.text(f"Extracting: {idx + 1}/{total_files} files")
        
        extract_progress.empty()
        status_text.empty()
        
        # Clean up zip file
        os.remove(MODEL_ZIP_NAME)
        
        st.success("✅ Model extracted successfully!")
        return True
        
    except zipfile.BadZipFile:
        st.error("Error: Downloaded file is not a valid zip file")
        return False
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        return False

def download_model_from_github():
    """
    Main function to download model from GitHub Release
    Handles both single file and chunked downloads
    
    Returns:
        bool: True if model is ready, False otherwise
    """
    
    # Check if model already exists locally
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
        st.success("✅ Model already exists locally!")
        return True
    
    st.warning("🔄 Model not found locally. Downloading from GitHub Release...")
    st.info(f"📍 Repository: {GITHUB_REPO}\n📌 Release: {MODEL_RELEASE_TAG}")
    
    # Create expander for download progress
    with st.expander("📊 Download Progress", expanded=True):
        
        # Check if model is chunked
        is_chunked, metadata = check_if_chunked()
        
        if is_chunked:
            # Download and merge chunks
            st.info(f"Model is split into {metadata['total_chunks']} chunks "
                   f"({metadata['total_size_bytes'] / (1024**3):.2f} GB total)")
            
            success = download_and_merge_chunks(metadata)
        else:
            # Download single file
            st.info("Downloading single model file...")
            success = download_single_file()
        
        if not success:
            st.error("❌ Download failed!")
            return False
        
        # Extract the zip file
        if not extract_model():
            st.error("❌ Extraction failed!")
            return False
        
        # Verify extraction
        if not os.path.exists(MODEL_DIR):
            st.error(f"❌ Model directory not found after extraction: {MODEL_DIR}")
            return False
        
        if not os.path.exists(os.path.join(MODEL_DIR, "adapter_config.json")):
            st.error("❌ Model files incomplete. Missing adapter_config.json")
            return False
    
    st.success("🎉 Model downloaded and ready!")
    return True

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load model from local directory
    Uses caching to avoid reloading on every interaction
    
    Returns:
        tuple: (model, tokenizer, error_message)
    """
    
    try:
        # Verify model directory exists
        if not os.path.exists(MODEL_DIR):
            return None, None, f"Model directory not found: {MODEL_DIR}"
        
        # Load PEFT configuration
        config_path = os.path.join(MODEL_DIR, "adapter_config.json")
        if not os.path.exists(config_path):
            return None, None, "adapter_config.json not found in model directory"
        
        config = PeftConfig.from_pretrained(MODEL_DIR)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        model.eval()
        
        return model, tokenizer, None
        
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# CODE GENERATION
# ============================================================================

def generate_code(model, tokenizer, pseudo_code, temperature=0.3, max_tokens=256, top_p=0.9):
    """
    Generate C++ code from pseudo-code
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        pseudo_code: Input pseudo-code string
        temperature: Sampling temperature (0.1-1.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        
    Returns:
        str: Generated C++ code
    """
    
    if not pseudo_code.strip():
        return "// Please enter pseudo-code to generate C++ code"
    
    # Create prompt
    prompt = f"Pseudo-code:\n{pseudo_code}\n\nC++ Code:\n"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate code
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode output
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated code part
        if prompt in generated:
            code = generated.split(prompt)[-1].strip()
        elif "C++ Code:" in generated:
            code = generated.split("C++ Code:")[-1].strip()
        else:
            code = generated.strip()
        
        # Validate output
        if not code or code == prompt:
            return "// Generation failed. Please try different pseudo-code or adjust settings."
        
        return code
        
    except Exception as e:
        return f"// Error during generation: {str(e)}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🚀 SPoC Code Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Convert pseudo-code into C++ code using AI</div>',
        unsafe_allow_html=True
    )
    
    # Step 1: Download model if needed
    st.info("🔍 Checking for model files...")
    
    if not download_model_from_github():
        st.error("❌ Failed to download model from GitHub Release")
        
        # Show detailed troubleshooting
        st.markdown(f"""
        ### 🔧 Troubleshooting Steps
        
        **Verify these items:**
        
        1. **Release exists**: Check [this link](https://github.com/{GITHUB_REPO}/releases/tag/{MODEL_RELEASE_TAG})
        2. **File uploaded**: Look for `{MODEL_ZIP_NAME}` in the release assets
        3. **Release is public**: Ensure the release is not in draft mode
        4. **Repository name**: Verify `{GITHUB_REPO}` is correct
        
        **Common Error Messages:**
        - `404 Not Found` → Release or file doesn't exist
        - `403 Forbidden` → Release is private or repository doesn't exist
        - `Timeout` → File is too large or network issue
        
        **How to fix:**
        1. Go to your GitHub repository
        2. Click "Releases" → "Create a new release"
        3. Tag version: `{MODEL_RELEASE_TAG}`
        4. Upload your model zip file
        5. Click "Publish release"
        6. Refresh this page
        
        **Need help?** Check the README or open an issue on GitHub.
        """)
        
        return
    
    # Step 2: Load model
    with st.spinner("🤖 Loading AI model into memory..."):
        model, tokenizer, error = load_model()
    
    if error:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Try refreshing the page. If the error persists, the model files may be corrupted.")
        return
    
    st.success("✅ Model loaded and ready to generate code!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values make output more creative/random. Lower values make it more deterministic."
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=512,
            value=256,
            step=32,
            help="Maximum length of the generated code"
        )
        
        top_p = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Sample from top P probability mass. Lower values = more focused output."
        )
        
        st.markdown("---")
        
        # Examples dropdown
        st.header("📖 Example Inputs")
        examples = {
            "": "Select an example...",
            "Simple Loop": "for i in range(0, 10): print i",
            "Conditional": "if x > 5: return true else return false",
            "Function": "function add(a, b): return a + b",
            "While Loop": "while condition is true: do something",
            "Factorial": "calculate factorial of n using recursion",
            "Max of Two": "find maximum of two numbers a and b",
            "Sort Array": "read array of n integers and sort it in ascending order",
            "Swap Variables": "swap two variables a and b without using temporary variable",
            "Sum Array": "calculate sum of all elements in array",
            "Binary Search": "perform binary search on sorted array to find element x"
        }
        
        selected_example = st.selectbox(
            "Quick start examples:",
            list(examples.keys()),
            format_func=lambda x: examples[x] if x else "Select an example..."
        )
        
        st.markdown("---")
        
        # Model information
        st.header("ℹ️ Model Info")
        st.markdown(f"""
        **Base Model:** distilgpt2  
        **Fine-tuned on:** SPoC Dataset  
        **Method:** PEFT (LoRA)  
        **Repository:** [{GITHUB_REPO}](https://github.com/{GITHUB_REPO})
        """)
        
        # Device information
        device = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Running on: {device}")
        
        # Model size info
        if os.path.exists(MODEL_DIR):
            model_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(MODEL_DIR)
                for filename in filenames
            ) / (1024 * 1024)
            st.caption(f"Model size: {model_size:.1f} MB")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Input: Pseudo-code")
        
        # Get example text if selected
        default_text = examples.get(selected_example, "") if selected_example else ""
        
        pseudo_code = st.text_area(
            "Enter your pseudo-code:",
            value=default_text,
            height=350,
            placeholder="Example:\nfor i in range(0, 10):\n    print i\n\nTip: Be clear and specific for best results!",
            help="Write your pseudo-code here. Use simple, clear statements for best results."
        )
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            generate_btn = st.button(
                "🔄 Generate C++ Code",
                type="primary",
                use_container_width=True
            )
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
        
        with col_btn3:
            st.button("📋 Copy", use_container_width=True, disabled=True, help="Use copy button below generated code")
    
    with col2:
        st.subheader("💻 Output: C++ Code")
        
        # Generate code if button clicked
        if generate_btn:
            if not pseudo_code.strip():
                st.warning("⚠️ Please enter some pseudo-code first!")
            else:
                with st.spinner("🔄 Generating C++ code... This may take a few seconds..."):
                    cpp_code = generate_code(
                        model,
                        tokenizer,
                        pseudo_code,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                
                # Display generated code
                st.code(cpp_code, language="cpp", line_numbers=True)
                
                # Download button
                st.download_button(
                    label="📥 Download C++ File",
                    data=cpp_code,
                    file_name="generated_code.cpp",
                    mime="text/x-c++src",
                    use_container_width=True
                )
                
                # Show statistics and feedback
                if not cpp_code.startswith("//"):
                    st.success("✅ Code generated successfully!")
                    
                    # Code statistics
                    lines = len(cpp_code.split('\n'))
                    chars = len(cpp_code)
                    words = len(cpp_code.split())
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Lines", lines)
                    with col_stat2:
                        st.metric("Characters", chars)
                    with col_stat3:
                        st.metric("Words", words)
                else:
                    st.info("ℹ️ Generation completed. Review the output above.")
        else:
            # Show placeholder when no generation yet
            st.info("👈 Enter pseudo-code and click 'Generate' to see C++ code")
            
            # Show example output
            st.code("""// Generated C++ code will appear here
//
// Example transformation:
// 
// Input (Pseudo-code):
//   for i in range(0, 10): print i
//
// Output (C++ Code):
//
#include <iostream>
using namespace std;

int main() {
    for(int i = 0; i < 10; i++) {
        cout << i << endl;
    }
    return 0;
}""", language="cpp")
    
    # Footer section
    st.markdown("---")
    
    # Three-column footer
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**📊 Dataset**")
        st.caption("SPoC (Semantic Parsing of Code)")
    
    with col_f2:
        st.markdown("**🔧 Framework**")
        st.caption("Transformers + PEFT (LoRA)")
    
    with col_f3:
        st.markdown("**📝 License**")
        st.caption("MIT License")
    
    # Expandable information sections
    with st.expander("⚠️ Limitations & Disclaimers"):
        st.markdown("""
        - **Best for simple code**: Works well with loops, conditionals, basic functions
        - **May need adjustment**: Generated code might require minor modifications
        - **Not production-ready**: Always review and test generated code
        - **Complex algorithms**: May not fully capture intricate logic
        - **No compilation guarantee**: Code may have syntax errors in some cases
        - **Educational purpose**: Use as a learning tool, not for critical applications
        """)
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        1. **Be specific**: Clearly state what you want (e.g., "sort array in ascending order")
        2. **Use standard constructs**: Stick to common patterns (for, while, if-else)
        3. **Mention types**: Specify data types when relevant (integer, string, array)
        4. **One task at a time**: Break complex problems into smaller parts
        5. **Iterate**: If output isn't good, rephrase your pseudo-code
        6. **Adjust temperature**: Lower (0.1-0.3) for deterministic, higher (0.5-0.8) for creative
        7. **Test output**: Always compile and test the generated code
        """)
    
    with st.expander("📚 Example Transformations"):
        st.markdown("""
        **Example 1: Simple Loop**
        ```
        Input:  for i in range(0, 5): print i
        Output: for(int i = 0; i < 5; i++) { cout << i << endl; }
        ```
        
        **Example 2: Conditional Statement**
        ```
        Input:  if x > 10: return true else return false
        Output: if(x > 10) { return true; } else { return false; }
        ```
        
        **Example 3: Function Definition**
        ```
        Input:  function factorial(n): if n <= 1: return 1 else return n * factorial(n-1)
        Output: int factorial(int n) {
                    if(n <= 1) return 1;
                    return n * factorial(n-1);
                }
        ```
        
        **Example 4: Array Operations**
        ```
        Input:  read n numbers into array and find their sum
        Output: int arr[n], sum = 0;
                for(int i = 0; i < n; i++) {
                    cin >> arr[i];
                    sum += arr[i];
                }
        ```
        """)
    
    with st.expander("🔮 Future Improvements"):
        st.markdown("""
        - Support for more programming languages (Python, Java, JavaScript)
        - Code explanation and documentation generation
        - Syntax error detection and correction
        - Code optimization suggestions
        - Interactive debugging mode
        - Multi-step code generation for complex algorithms
        - Unit test generation
        """)
    
    with st.expander("📞 Support & Feedback"):
        st.markdown(f"""
        **Found a bug?** Open an issue on [GitHub](https://github.com/{GITHUB_REPO}/issues)
        
        **Want to contribute?** Pull requests are welcome!
        
        **Have feedback?** We'd love to hear from you:
        - Star the repository if you find it useful ⭐
        - Share with others who might benefit 🔄
        - Report issues or suggest features 💡
        
        **Credits:**
        - SPoC Dataset: [Kulal et al., 2019]
        - Hugging Face Transformers
        - Streamlit Framework
        """)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()