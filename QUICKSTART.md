# Quick Start Guide

This guide will help you get started with Finance Agent in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- NVIDIA GPU (recommended, but CPU works too)
- Git

## Installation Steps

### 1. Clone and Setup

```bash
cd financial-agent
```

### 2. Install Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using uv (faster)**
```bash
# Install uv if you haven't
pip install uv

# Install dependencies
uv sync
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your favorite editor
```

Required variables:
- `HF_TOKEN`: Get from [Hugging Face](https://huggingface.co/settings/tokens)
- `SEC_EMAIL`: Your email (required by SEC for downloads)
- `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com/) (optional, for web search)

### 4. Verify Setup

```bash
python setup_check.py
```

This will check all dependencies and configurations.

### 5. Download Financial Data

```bash
# This will download SEC filings and create a vector database
# Takes 5-10 minutes depending on your internet speed
python src/ingestion.py
```

### 6. Run the Agent

**Interactive Mode:**
```bash
python main.py
```

**Single Query:**
```bash
python main.py --query "What was Apple's revenue in 2023?"
```

**RAG Search Only:**
```bash
python main.py --rag-only --query "risk factors"
```

## Example Queries

Once the agent is running, try these questions:

1. **Financial Data Queries:**
   - "What was Apple's total revenue in the latest fiscal year?"
   - "Compare Microsoft and Tesla's operating margins"
   - "What are the main risk factors for AAPL?"

2. **Calculations:**
   - "Calculate the year-over-year growth if revenue was $100M and is now $125M"
   - "If I invest $10,000 with 8% annual return for 5 years, how much will I have?"

3. **Current Information (requires Tavily API):**
   - "What is the current stock price of Apple?"
   - "Latest news about Tesla earnings"

## Project Structure

```
financial-agent/
├── config/              # Configuration files
├── data/                # Data storage (created after ingestion)
│   ├── sec_filings/    # Downloaded SEC reports
│   └── vector_db/      # Vector database
├── src/
│   ├── ingestion.py    # Download & process SEC filings
│   ├── rag_chain.py    # RAG pipeline with reranking
│   ├── tools.py        # Agent tools (search, calculator, etc.)
│   ├── agent.py        # Main agent logic
│   └── train.py        # Fine-tuning script
├── main.py             # CLI entry point
├── setup_check.py      # Environment checker
└── README.md           # Full documentation
```

## Common Issues

### Issue: "CUDA out of memory"
**Solution:** Use a smaller model or CPU
```bash
python main.py --model Qwen/Qwen2.5-3B-Instruct --device cpu
```

### Issue: "Vector database not found"
**Solution:** Run ingestion first
```bash
python src/ingestion.py
```

### Issue: "HF_TOKEN not configured"
**Solution:** Add your token to .env
```bash
# Get token from https://huggingface.co/settings/tokens
echo "HF_TOKEN=hf_your_token_here" >> .env
```

## Next Steps

1. **Explore the Tools:**
   ```bash
   python src/tools.py  # Test individual tools
   ```

2. **Test RAG Pipeline:**
   ```bash
   python src/rag_chain.py  # Test retrieval
   ```

3. **Fine-tune Model (Advanced):**
   ```bash
   # Create training dataset
   python src/train.py --create-dataset

   # Start fine-tuning (requires GPU)
   python src/train.py --train
   ```

4. **Customize Configuration:**
   - Edit `config/model_config.py` for model settings
   - Modify RAG parameters in the config files
   - Add more tools in `src/tools.py`

## Getting Help

- Run `python main.py --help` for CLI options
- Run `python setup_check.py` for diagnostics
- Check `README.md` for detailed documentation
- Open an issue on GitHub if you encounter problems

## Performance Tips

1. **Use GPU:** CUDA significantly speeds up inference
2. **Adjust batch size:** Reduce if you get OOM errors
3. **Cache models:** Models are cached after first download
4. **Use quantization:** 4-bit models use less memory

Enjoy using Finance Agent! 🚀
