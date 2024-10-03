


from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForQuestionAnswering.from_pretrained(model_name)

question_answering = pipeline("question-answering", model=model, tokenizer=tokenizer)

"""-  we read the model
- we read the tokenizer
- we set up our pipeline      
 like this we done the sentimental analysis
"""

# input passage and quation:
passage =("Konidala Pawan Kalyan (born 2 September 1971)[2] is an Indian politician, actor and philanthropist, serving as the 10th Deputy Chief Minister of Andhra Pradesh since June 2024. He is also the Minister of Panchayat Raj, Rural Development and Rural Water Supply; Environment, Forests, Science and Technology in the Government of Andhra Pradesh and an MLA representing the Pitapuram constituency.[3] He is the founder and president of the Janasena Party.")

quation = "who is pawan kalyan ?"

# Perform Question and answering
result = question_answering(question=quation, context=passage)

# print result
print(f"Question: {quation}")
print(f"Answer: {result['answer']}")

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

#Load pretrained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

question_answering = pipeline("question-answering", model=model, tokenizer=tokenizer)