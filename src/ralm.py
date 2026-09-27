import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class InContextRALM:

    def __init__(self, model_name="google/flan-t5-small"):

        print(f"Loading model: {model_name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded.")

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    def generate_baseline(self, question):

        prompt = (
            "Answer the following question.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return self._generate(prompt)

    # ---------------------------------------------------------
    # IN-CONTEXT RALM
    # ---------------------------------------------------------

    def generate_ralm(self, question, documents):

        # Keep retrieved documents short enough that the
        # question is never truncated.
        max_doc_tokens = 70

        processed_documents = []

        for i, doc in enumerate(documents):

            doc_tokens = self.tokenizer.encode(
                doc,
                add_special_tokens=False
            )

            doc_tokens = doc_tokens[:max_doc_tokens]

            truncated_doc = self.tokenizer.decode(
                doc_tokens,
                skip_special_tokens=True
            )

            processed_documents.append(
                f"Document {i + 1}: {truncated_doc}"
            )

        context = "\n\n".join(processed_documents)

        prompt = (
            "Answer the question using the provided documents.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return self._generate(prompt)

    # ---------------------------------------------------------
    # GENERATION
    # ---------------------------------------------------------

    def _generate(self, prompt):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=2
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()