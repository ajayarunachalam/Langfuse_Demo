import ollama
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from langfuse import Langfuse

# Load environment variables
load_dotenv()

# Initialize Langfuse
langfuse = Langfuse()
trace_id = str(uuid.uuid4())
ollama_trace = langfuse.trace(id=trace_id, name="ollama_content_trace")

# Prompt the user for the topic
user_topic = input("Please enter the topic for content generation: ")

# Instruction prompt for content generation
content_generation_prompt = f"Generate detailed content on the following topic:\n{user_topic}\nEnsure the content is comprehensive and well-structured."

# Trace the content generation
content_generation = ollama_trace.generation(
    name="content_generation",
    input={
        "prompt": user_topic,
        "instruction_prompt": content_generation_prompt,
    },
    metadata={"version": 1}
)

# Generate content using Ollama model
content_output = ollama.generate("llama3.2:latest", prompt=content_generation_prompt)
content_generation.end(
    output=content_output.get("response"),
)


# Capture user feedback for the generated content before displaying it
print("Generated Content (Preview):")
print(content_output.get("response"))
content_feedback = input("Please provide feedback for the generated content (good, bad, worst): ")
content_generation.update(metadata={"user_feedback": content_feedback})

# Instruction prompt for summarization
summarization_prompt = f"Summarize the following content:\n{content_output.get('response')}\nEnsure the summary is concise and captures the key points."

# Trace the summarization
summarization_generation = ollama_trace.generation(
    name="summarization_generation",
    input={
        "prompt": content_output.get("response"),
        "instruction_prompt": summarization_prompt,
    },
    metadata={"version": 1}
)

# Generate summary using Ollama model
summary_output = ollama.generate("llama3.2:latest", prompt=summarization_prompt)
summarization_generation.end(
    output=summary_output.get("response"),
)

# Capture user feedback for the generated summary before displaying it
print("Generated Summary (Preview):")
print(summary_output.get("response"))
summary_feedback = input("Please provide feedback for the generated summary (good, bad, worst): ")
summarization_generation.update(metadata={"user_feedback": summary_feedback})

# Update the trace with the final output
ollama_trace.update(input=user_topic, output=summary_output)

# Evaluate the output using LLM-as-judge evaluators
evaluation_prompt = f"Evaluate the following content and summary:\nContent:\n{content_output.get('response')}\nSummary:\n{summary_output.get('response')}\nProvide a score out of 10 for accuracy, coherence, and relevance."

# Trace the evaluation
evaluation_generation = ollama_trace.generation(
    name="evaluation_generation",
    input={
        "prompt": evaluation_prompt,
    },
    metadata={"version": 1}
)

# Generate evaluation using Ollama model
evaluation_output = ollama.generate("llama3.2:latest", prompt=evaluation_prompt)
evaluation_generation.end(
    output=evaluation_output.get("response"),
)

# Capture user feedback for the evaluation before displaying it
print("Generated Evaluation (Preview):")
print(evaluation_output.get("response"))
evaluation_feedback = input("Please provide feedback for the evaluation (good, bad, worst): ")
evaluation_generation.update(metadata={"user_feedback": evaluation_feedback})

# Update the trace with the evaluation output
ollama_trace.update(input=evaluation_prompt, output=evaluation_output)

# Print the final outputs
print("Final Generated Content:")
print("Generated Content:", content_output.get("response"))
print("Summary:", summary_output.get("response"))
print("Evaluation:", evaluation_output.get("response"))


