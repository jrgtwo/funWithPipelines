from InquirerPy import prompt

def selectTask():
    tasks = [
        'audio-classification', 
        'automatic-speech-recognition', 
        'depth-estimation', 
        'document-question-answering', 
        'feature-extraction', 
        'fill-mask', 
        'image-classification', 
        'image-feature-extraction', 
        'image-segmentation', 
        'image-text-to-text', 
        'image-to-image', 
        'image-to-text', 
        'keypoint-matching', 
        'mask-generation', 
        'ner', 
        'object-detection', 
        'question-answering', 
        'sentiment-analysis', 
        'summarization', 
        'table-question-answering', 
        'text-classification', 
        'text-generation', 
        'text-to-audio', 
        'text-to-speech', 
        'text2text-generation', 
        'token-classification', 
        'translation', 
        'video-classification', 
        'visual-question-answering', 
        'vqa', 
        'zero-shot-audio-classification', 
        'zero-shot-classification', 
        'zero-shot-image-classification', 
        'zero-shot-object-detection', 
        'translation_XX_to_YY'
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