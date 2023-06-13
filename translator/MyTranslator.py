from translate import Translator
from langdetect import detect

class MyTranslator:
    def __init__(self):
        self.text = ''
        # destiny language
        self.to_lang = 'en'
        self.from_lang = ''
        self.translator = None
    
    def detect_language(self, text):
        detected_lang = detect(text)
        self.from_lang = detected_lang

    def translate(self, text):
        self.translator = Translator(to_lang=self.to_lang, from_lang=self.from_lang)
        self.text = text
        translation = self.translator.translate(text)
        return translation
         
