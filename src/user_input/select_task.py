from InquirerPy import prompt

def selectTask():
    tasks = [
        "text-generation",
        "text-classification",
        "question-answering",
        "summarization",
        "translation",
    ]

    questions = [
        {
            "type": "list",
            "name": "task",
            "message": "Select the NLP task:",
            "choices": tasks,
        }
    ]

    answers = prompt(questions)
    return answers["task"]