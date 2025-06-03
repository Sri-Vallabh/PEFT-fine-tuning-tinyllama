
# Parameter-Efficient Fine-Tuning of TinyLlama with QLoRA

## Project Overview
This project explores parameter-efficient fine-tuning (PEFT) of the **TinyLlama-1.1B-Chat** model using **QLoRA** (Quantized Low-Rank Adaptation). The goal is twofold:
1. **Adapt the base model to specialized philosophical domains** (abduction, Abelard, and Abhidharma) while minimizing computational costs.
2. **Optimize fine-tuning time** by leveraging efficient hardware and best practices, running on a consumer-grade PC with an **RTX 4050 (6GB VRAM)**.

The workflow involves:
- **Full-dataset fine-tuning**
- **Category-specific adapter training**
- **Adapter merging via multiple methods**
- **Evaluation using BERTScore and LLM-as-a-judge (Llama 3 8B-8192)**

---

## Dataset
- **Size**: 133,000 samples across three philosophical categories:
  - **Abduction** (Logical reasoning)
  - **Abelard** (Medieval philosophy)
  - **Abhidharma** (Buddhist philosophy)
  
    ...so on
- **Structure**: Each sample contains:
  ```
  {"question": "...", "answer": "...", "category": "abduction|abelard|abhidharma|..."}
  ```

---

## Methodology

### Phase 1: Full-Dataset Fine-Tuning
- **Objective**: Establish baseline performance and check hardware feasibility
- **Configuration**:
  - **QLoRA**: 4-bit quantization with NF4 type
  - **LoRA Rank**: r = 64
  - **Training**:
    - 3 epochs (17 hours on RTX 4050 6GB)
    - Batch size = 4, gradient accumulation = 2
    - Learning rate = 5e-4 (cosine scheduler)
  - **Optimization**: Focus on maximizing throughput and minimizing VRAM usage for fast convergence on consumer hardware[2][3].

### Phase 2: Category-Specific Adapters
- **Strategy**: Split dataset by category and train separate adapters:
  - `final_abduction_adapter`
  - `final_abelard_adapter`
  - `final_abhidharma_adapter`
- **Training**:
  - 24 epochs per category(~10 minutes)
  - Same QLoRA config as Phase 1
- **Goal**: Reduce training time per category by leveraging efficient batch packing that is also within gpu limits[2][3].

---

## Adapter Merging Techniques
Multiple merging strategies were implemented and compared:

| Method              | Key Parameters               | Description                                  |
|---------------------|------------------------------|----------------------------------------------|
| **Linear**          | `weights=[1.0,1.0,1.0]`      | Weighted average of adapter parameters       |
| **TIES**            | `density=0.5`                | Resolves parameter conflicts via pruning     |
| **DARE**            | `density=0.7`                | Drops random parameters + rescaling          |
| **Magnitude Prune** | `density=0.9`                | Removes smallest-magnitude parameters        |
| **SVD**             | `density=0.01` | Rank reduction via singular value decomposition |


**Example Merge Code**:
```
model.add_weighted_adapter(
    adapters=["abduction", "abelard", "abhidharma"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="merged",
    combination_type="ties",
    density=0.5
)
```

---

## Evaluation Framework

### 1. BERTScore Metrics
- **Metrics**:
  - **Precision**: Token-level relevance to reference
  - **Recall**: Coverage of reference content
  - **F1**: Balance of precision/recall
- **Sample Scores**:
  ```
  Base Model:       F1=0.72
  Full Fine-Tuned:  F1=0.85
  Merged Adapters:  F1=0.88
  ```

### 2. LLM-as-Judge (Llama 3 8B-8192)
- **Method**:
  - Compare base vs. fine-tuned responses using:
    ```
    def llm_judge(question, reference, base_gen, fine_tuned_gen):
    """Get LLM judgment between base and fine-tuned answers"""
    prompt = f"""Compare reference answer to the base model answer and fine-tuned answer and identify which is more appropriate answer, the base model answer or fine-tuned answer. Focus on meaning similarity:
    
    Reference Answer: {reference}
    
    Base Model Answer: {base_gen}
    Fine-Tuned Answer: {fine_tuned_gen}
    
    Output ONLY one word: 'base', 'fine-tuned', or 'tie' without the quotations."""
    response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a semantic similarity expert"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().lower()
    ```
- **Result**: Fine-tuned model preferred in **193/194** of cases

---

## Key Findings
1. **Adapter Merging**:
   - **TIES** (density=0.5) performed best for conflicting categories
   - **Linear** merging worked well for complementary domains
2. **Efficiency**:
   - **QLoRA reduced memory usage by 4×** compared to full fine-tuning, enabling training on RTX 4050 6GB VRAM[2][3]
   - **Optimized batch packing and dynamic batching** helped maximize throughput and minimize training time
3. **Specialization**:
   - Category-specific adapters showed **15% accuracy improvement** over base

---

## Repository Structure
```
├── /adapters/
│   ├── final_abduction_adapter/      # Category-specific adapter: Abduction
│   ├── final_abelard_adapter/        # Category-specific adapter: Abelard
│   └── final_abhidharma_adapter/     # Category-specific adapter: Abhidharma
├── /merged_adapters/
│   ├── linear_merged_adapter/        # Linear merged adapter
│   └── magnitude_prune_merged_adapter/  # Magnitude-pruned merged adapter
├── philosophy_qa_fixed.json          # Raw dataset (133k samples)
├── base_abduction_results.csv        # Base model results (Abduction)
├── base_abduction_results_l.csv      # (Optional) Base model results, variant
├── base_merged_results.csv           # Base model results (Merged categories)
├── fine_tuned_abduction_results.csv  # Fine-tuned results (Abduction)
├── fine_tuned_merged_results.csv     # Fine-tuned results (Merged categories)
├── fine_tuned_merged_linear_results.csv  # Fine-tuned results (Linear merged)
├── Ilm_semantic_merged_judgments.csv # LLM-as-judge 
├── .gitattributes                    # Git configuration
├── README.md                         # Project documentation
├── philosophy.ipynb                  # Notebook: Full-dataset fine-tuning (3 epochs)
├── category.ipynb                    # Notebook: Category-wise fine-tuning (24 epochs)
├── testing.ipynb                     # Notebook: Adapter merging, BERTScore, LLM-as-judge
├── app.py                            # Streamlit app: Compare fine-tuned vs original results


```

---

## Getting Started
1. **Installation**:
   ```
   pip install peft transformers datasets accelerate
   ```
2. **Load Merged Adapter**:
   ```
   model = PeftModel.from_pretrained(
       base_model, 
       "./merged_adapters/ties_merge"
   )
   ```

---

## Future Work

- Explore **cross-category attention** mechanisms
- Expand to 10+ philosophical categories

---

