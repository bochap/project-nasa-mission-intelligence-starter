# NASA RAG Chat Project - Student Learning Version

A hands-on learning project for building a Retrieval-Augmented Generation (RAG) system with real-time evaluation capabilities. This project teaches students to create a complete RAG pipeline from document processing to interactive chat interface.

## 🎯 Learning Objectives

By completing this project, students will learn to:
- Build document embedding pipelines with ChromaDB and OpenAI
- Implement RAG retrieval systems with semantic search
- Create LLM client integrations with conversation management
- Develop real-time evaluation systems using RAGAS metrics
- Build interactive chat interfaces with Streamlit
- Handle error scenarios and edge cases in production systems

## 📁 Project Structure

```
/
├── chat.py                 # Main Streamlit chat application (TODO-based)
├── embedding_pipeline.py   # ChromaDB embedding pipeline (TODO-based)
├── llm_client.py           # OpenAI LLM client wrapper (TODO-based)
├── rag_client.py           # RAG system client (TODO-based)
├── ragas_evaluator.py      # RAGAS evaluation metrics (TODO-based)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key
- Basic understanding of Python, APIs, and vector databases
- Familiarity with machine learning concepts

### Installation & Environment Setup

#### Option A: Running with `uv` (Recommended)
`uv` is an extremely fast package installer and resolver.
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Sync dependencies from requirements.txt
uv pip install -r requirements.txt
```

[**Note** To install uv, please refer to the official uv documentation](https://astral.sh/uv/docs/installation/).

#### Option B: Running without `uv` (Standard `pip`)
If you do not have `uv` installed, use standard Python virtual environments:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Configuration

Configure your environment variables by creating a `.env` file in the project root folder (you can copy or edit the provided `.env` template):
```env
OPENAI_API_KEY="your-openai-api-key"
NASA_LOG_LEVEL="INFO"
```
Alternatively, you can export it directly in your terminal session:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 📋 Project Rubric & Compliance

The implementation is completed based on the rubric in [rubric.md](rubric.md). The rubric also contains a complete mapping under the **Code References & Implementation** column linking each requirement directly to the specific files, class names, and line ranges in the codebase.

## 📚 Learning Path

This project follows a structured learning approach where each file contains TODO comments guiding you through the implementation. Complete the files in this recommended order:

### **Phase 1: Core Infrastructure**

#### 1. **LLM Client (`llm_client.py`)** - *Estimated Time: 2-3 hours*
**What you'll learn:**
- OpenAI Chat Completions API integration
- System prompt engineering for domain expertise
- Conversation history management
- Context integration strategies
- Model parameter tuning (temperature, max_tokens)

**Key TODOs:**
- Define system prompt for NASA expertise
- Set context in messages
- Add chat history management
- Create OpenAI Client
- Send request to OpenAI and return response

#### 2. **RAG Client (`rag_client.py`)** - *Estimated Time: 3-4 hours*
**What you'll learn:**
- ChromaDB backend discovery and connection
- Semantic search with metadata filtering
- Document retrieval optimization
- Context formatting for LLM consumption

**Key TODOs:**
- Discover available ChromaDB collections
- Initialize RAG system with database connections
- Implement document retrieval with optional filtering
- Format retrieved documents into structured context

#### 3. **Embedding Pipeline (`embedding_pipeline.py`)** - *Estimated Time: 6-8 hours*
**What you'll learn:**
- Document processing and text chunking strategies
- OpenAI embeddings generation
- ChromaDB collection management
- Metadata extraction and organization
- Batch processing and error handling
- Command-line interface development

**Key TODOs:**
- Initialize OpenAI client and ChromaDB
- Implement intelligent text chunking with overlap
- Create document management methods
- Build metadata extraction from file paths
- Implement batch document processing
- Create command-line interface

### **Phase 2: Evaluation and Interface**

#### 4. **RAGAS Evaluator (`ragas_evaluator.py`)** - *Estimated Time: 2-3 hours*
**What you'll learn:**
- Response quality evaluation metrics
- RAGAS framework integration
- Multi-dimensional assessment (relevancy, faithfulness, precision)
- Evaluation data structure management

**Key TODOs:**
- Create evaluator LLM and embeddings
- Define evaluation metrics instances
- Evaluate responses using multiple metrics
- Return comprehensive evaluation results

#### 5. **Chat Application (`chat.py`)** - *Estimated Time: 4-5 hours*
**What you'll learn:**
- Streamlit web application development
- Real-time evaluation integration
- User interface design for RAG systems
- Session state management
- Configuration and settings management

**Key TODOs:**
- Integrate all components (RAG, LLM, evaluation)
- Build interactive chat interface
- Implement real-time quality metrics display
- Handle user configuration and backend selection

## 🛠️ Implementation Guidelines

### **TODO-Based Learning Approach**
Each file contains strategically placed TODO comments that guide you through:
1. **Understanding the purpose** of each function/method
2. **Implementing core logic** step by step
3. **Handling edge cases** and error scenarios
4. **Integrating components** effectively

### **Code Quality Standards**
- Follow Python PEP 8 style guidelines
- Add comprehensive error handling
- Include informative logging statements
- Write clear docstrings for all functions
- Use type hints for better code clarity

### **Testing Strategy**
- Test each component individually before integration
- Use small datasets for initial testing
- Verify API connections before processing large batches
- Test edge cases (empty files, network errors, invalid inputs)

## 📊 Data Requirements

### **Expected Data Structure**
The system expects NASA document data organized in folders:
```
data/
├── apollo11/           # Apollo 11 mission documents
│   ├── *.txt          # Text files with mission data
├── apollo13/           # Apollo 13 mission documents
│   ├── *.txt          # Text files with mission data
└── challenger/         # Challenger mission documents
    ├── *.txt          # Text files with mission data
```

### **Supported Document Types**
- Plain text files (.txt)
- Mission transcripts
- Technical documents
- Audio transcriptions
- Flight plans and procedures

## 🧪 Testing Your Implementation

### **Component Testing**

1. **Test LLM Client**:
   ```python
   from llm_client import generate_response
   response = generate_response(api_key, "What was Apollo 11?", "", [])
   print(response)
   ```

2. **Test RAG Client**:
   ```python
   from rag_client import discover_chroma_backends
   backends = discover_chroma_backends()
   print(backends)
   ```

3. **Test Embedding Pipeline**:
   With `uv`:
   ```bash
   uv run embedding_pipeline.py --stats-only
   ```
   Without `uv`:
   ```bash
   python embedding_pipeline.py --stats-only
   ```

4. **Test Evaluation**:
   ```python
   from ragas_evaluator import evaluate_response_quality
   scores = evaluate_response_quality("question", "answer", ["context"])
   print(scores)
   ```

### **Integration Testing**

1. **Run the complete pipeline**:
   *   **Option A: Using `uv`**:
       ```bash
       # Process documents
       uv run embedding_pipeline.py --data-path ./data_text
       
       # Launch chat interface
       uv run streamlit run chat.py
       ```
   *   **Option B: Without `uv`**:
       ```bash
       # Process documents
       python embedding_pipeline.py --data-path ./data_text
       
       # Launch chat interface
       streamlit run chat.py
       ```

2. **Run the comparative evaluation suite**:
   *   **Option A: Using `uv`**:
       ```bash
       uv run run_evaluation_dataset.py
       ```
   *   **Option B: Without `uv`**:
       ```bash
       python run_evaluation_dataset.py
       ```
   This loads test cases from `test_questions.json`, evaluates Filtered vs. Unfiltered modes, prints a comparative summary console table, and saves the report to `evaluation_comparison.md`.

## 🎓 Learning Checkpoints

### **Checkpoint 1: Basic Functionality**
- [ ] LLM client generates responses
- [ ] RAG client discovers ChromaDB backends
- [ ] Embedding pipeline processes sample files
- [ ] Evaluation system calculates basic metrics

### **Checkpoint 2: Integration**
- [ ] Components work together seamlessly
- [ ] Chat interface loads and responds to queries
- [ ] Real-time evaluation displays metrics
- [ ] Error handling works correctly

### **Checkpoint 3: Advanced Features**
- [ ] Mission-specific filtering works
- [ ] Conversation history is maintained
- [ ] Batch processing handles large datasets
- [ ] Performance is acceptable for interactive use

## 🚨 Common Challenges and Solutions

### **API Integration Issues**
- **Problem**: OpenAI API key errors
- **Solution**: Verify key is set correctly and has sufficient credits

### **ChromaDB Connection Issues**
- **Problem**: Collection not found errors
- **Solution**: Run embedding pipeline first to create collections

### **Memory and Performance Issues**
- **Problem**: Out of memory during processing
- **Solution**: Reduce batch sizes and chunk sizes

### **Evaluation Errors**
- **Problem**: RAGAS evaluation fails
- **Solution**: Ensure all dependencies are installed and contexts are properly formatted

## 📈 Success Metrics

Your implementation is successful when:
1. **Functionality**: All components work individually and together
2. **User Experience**: Chat interface is responsive and intuitive
3. **Quality**: Responses are relevant and well-sourced
4. **Evaluation**: Metrics provide meaningful quality assessment
5. **Robustness**: System handles errors gracefully
6. **Performance**: Response times are acceptable for interactive use

## 🔧 Configuration Options

### **Embedding Pipeline**
- Chunk size and overlap settings
- Batch processing parameters
- Update modes for existing documents
- Embedding model selection

### **LLM Client**
- Model selection (GPT-3.5-turbo, GPT-4)
- Temperature and creativity settings
- Maximum token limits
- Conversation history length

### **RAG System**
- Number of documents to retrieve
- Mission-specific filtering options
- Similarity thresholds

### **Evaluation System**
- Metric selection and weighting
- Evaluation frequency settings
- Display preferences

## 🏆 Extension Opportunities

Once you complete the basic implementation, consider these enhancements:

1. **Advanced Retrieval**: Implement hybrid search (semantic + keyword)
2. **Multi-modal Support**: Add support for images and audio
3. **Performance Optimization**: Add caching and parallel processing
4. **Advanced Evaluation**: Implement custom metrics for domain-specific quality
5. **Deployment**: Containerize and deploy to cloud platforms
6. **Monitoring**: Add comprehensive logging and monitoring
7. **Security**: Implement authentication and rate limiting

## 📚 Learning Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG System Design Patterns](https://python.langchain.com/docs/use_cases/question_answering/)

## 🤝 Getting Help

If you encounter issues:
1. Check the TODO comments for guidance
2. Review error messages carefully
3. Test components individually
4. Verify API keys and dependencies
5. Check data format and structure
6. Review the completed implementation in `project_completed/` folder

## 📋 Challenges Faced & Solutions Found

### 1. Document Chunking Strategy
*   **Challenge**: Finding an effective way to split document transcripts into clean chunks. I initially considered regex to locate sentence boundaries. After checking mentor answers (e.g. [Udacity Question 1085390](https://knowledge.udacity.com/questions/1085390)), using LLMs for chunking was a possibility, but I ruled it out due to high API costs.
*   **Solution**: I chose to use the `spaCy` NLP library for sentence boundary detection. However, parsing complete transcript files in one pass led to excessive memory consumption. To solve this, I introduced a localized windowing search that slices text segments to `2 * chunk_size` before running the parser.

### 2. ChromaDB Version Compatibility
*   **Challenge**: ChromaDB's frequent API updates make online documentation confusing, as many search results show obsolete code versions.
*   **Solution**: I inspected [provider/plugins/chromadb_provider.py](provider/plugins/chromadb_provider.py#L77-L88) and noticed that `OpenAIEmbeddingFunction` is registered directly on the Chroma collection. This makes calling `get_embedding()` manually in the pipeline technically redundant since ChromaDB can auto-vectorize. However, I decided to keep the explicit `get_embedding()` call for three reasons:
    *   **Rubric Compliance**: [rubic.md](rubic.md) requires explicitly calling an OpenAI embedding model to vectorize each chunk.
    *   **Error Isolation & Logging**: Allows us to intercept and log rate-limit or authentication exceptions before it hits database transactions.
    *   **Decoupled Architecture**: Keeps the pipeline compatible if the collection is initialized without a default embedding function.

## 🚀 Additional Features & Improvements

In addition to implementing the core requirements, I added several production-grade improvements to enhance the system's robustness, security, and verification capabilities:

### 1. Unified Environment Bootstrapping (`python-dotenv`)
*   Integrated `python-dotenv` and loaded it globally at the top of the [bootstrap.py](bootstrap.py) layer.
*   All downstream scripts automatically load settings from a `.env` template, eliminating the need to expose API keys in CLI arguments or commit secrets to git.

### 2. Dual-Mode Comparative RAG Evaluation
*   Modified [run_evaluation_dataset.py](run_evaluation_dataset.py) to execute each query under both **Filtered** (using mission-specific metadata) and **Unfiltered** (global search) modes.
*   Implemented metrics aggregation functions to compute summaries, delta changes, and percentage changes between modes. It outputs a formatted comparative console table and saves a Markdown report.

### 3. Out-of-Context RAG Grounding Verification
*   Added specific test questions in [test_questions.json](test_questions.json) targeting information absent from the NASA documents.
*   This explicitly tests and verifies that the LLM client correctly outputs `"cannot be found in the logs"` instead of generating hallucinations.

### 4. Extensible Provider Factory Pattern (Builder Concept)
*   Implemented a unified [provider/builder_registry.py](provider/builder_registry.py) that resolves LLM, RAG database, and Ragas metric dependencies at runtime.
*   Conforms to the abstract interfaces defined in [provider/protocol.py](provider/protocol.py) (e.g. `BaseBuilder`). 
*   Any new provider (e.g., Pinecone, Qdrant, or Claude) can be registered simply by writing a builder class and decorating it with `@BuilderRegistry().register_rag("provider_name")` or `@BuilderRegistry().register_llm("provider_name")`. This makes adding new engines straightforward when they share similar interaction contracts.

## 📝 Submission Guidelines

When submitting your completed project:
1. Ensure all TODO items are implemented
2. Test the complete workflow end-to-end
3. Include a brief report on challenges faced and solutions found
4. Document any additional features or improvements you added
5. Provide sample queries and expected responses

---

**Good luck with your RAG system implementation!** This project will give you hands-on experience with modern AI application development, from data processing to user interface design. Take your time with each component and don't hesitate to experiment with different approaches and parameters.
