from InquirerPy import prompt
from tasks.chat.personalities import chat_personalities

def select_persona():
    personas = chat_personalities()
    persona_list = personas.keys()
    questions = [
        {
            "type": "list",
            "name": "persona",
            "message": "Select chat persona:",
            "choices": persona_list,
        }
    ]

    answer = prompt(questions)["persona"]
    return personas[answer]