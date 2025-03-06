from enum import Enum
import os
import json
import shutil
from dotenv import load_dotenv
from glob import glob

import PyPDF2
from pdf2image import convert_from_path
import numpy as np  

from image_processing import extract_text_from_image

load_dotenv()


class PDFExtractionMethods(Enum):
    TEXT_THEN_LLM='text_then_llm'
    LLM='llm'


    @classmethod
    def list_modes(cls):
        return ', '.join([mode.name for mode in cls])
    

class PDFProcessor:
    def __init__(self, output_dir='', model='gpt-4o-mini', max_tokens=1000):
        
        self.output_dir=output_dir
        self.model=model
        self.max_tokens=max_tokens

    def process(self, path, method='llm', **kwargs):
        extraction_modes=PDFExtractionMethods.list_modes()
        if method not in extraction_modes:
            raise ValueError(f"Invalid extraction method: {method}. Valid methods are: {extraction_modes}")

        # Creating the pdf save directory
        filename=os.path.basename(path)
        filename=PDFProcessor.fix_filename(filename)

        # Setting up pdf save directory
        pdf_save_dir=os.path.join(self.output_dir,filename)
        pdf_image_dir=os.path.join(pdf_save_dir,'images')
        if os.path.exists(pdf_save_dir):
            return None

        # Extracting image from pdf
        if not os.path.exists(pdf_image_dir):
            PDFProcessor._extract_images_from_pdf(path=path, pdf_image_dir=pdf_image_dir, **kwargs)
        
        
        # Extarcting information from pdf
        if method == PDFExtractionMethods.LLM.value:
           pdf_dict=self._llm_processing(path=path, pdf_image_dir=pdf_image_dir)
        elif method == PDFExtractionMethods.TEXT_THEN_LLM.value:
            pdf_dict=self._text_then_llm_processing(path=path, pdf_image_dir=pdf_image_dir)

        self._save_dict(path=os.path.join(pdf_save_dir,'pdf_info.json'), pdf_dict=pdf_dict)
        return None
    
    def _llm_processing(self, path, pdf_image_dir):
        image_paths=glob(os.path.join(pdf_image_dir,'*.png'))

        # Creating dict to save processed information
        n_pages=len(image_paths)
        pdf_dict=self._construct_storage_dict(n_pages=n_pages)
        pdf_dict['metadata']['pdf_name']=os.path.basename(path).split('.')[0]
        pdf_dict['metadata']['pdf_path']=path

        for image_path in image_paths:
            image_name=os.path.basename(image_path)
            page_number=int(image_name.split('.')[0].split('_')[-1])
            prompt_and_response=extract_text_from_image(image_path, 
                                                    model=self.model,
                                                    max_tokens=self.max_tokens,
                                                    image_type='png')

            pdf_dict['pages'][f'page_{page_number}']['text']+=prompt_and_response[1]
            pdf_dict['metadata']['image_prompt']=prompt_and_response[0]
        return pdf_dict
    
    def _text_then_llm_processing(self, path, pdf_image_dir):
        pages_info=PDFProcessor._extract_information_from_pdf(path)

        # Creating dict to save processed information
        pdf_dict=PDFProcessor._construct_storage_dict(n_pages=pages_info['metadata']['num_pages'])
        pdf_dict['metadata']['pdf_name']=os.path.basename(path).split('.')[0]
        pdf_dict['metadata']['pdf_path']=path

        # Puting page information into the proper storage dict
        for key, page_dict in pages_info['pages'].items():
            pdf_dict['pages'][key]=page_dict

            if page_dict['process_as_image']:
                
                image_path=os.path.join(pdf_image_dir,f'{key}.png')
                prompt_and_response=extract_text_from_image(image_path, 
                                                    model=self.model,
                                                    max_tokens=self.max_tokens,
                                                    image_type='png')
                pdf_dict['pages'][key]['text']+=prompt_and_response[1]
                pdf_dict['metadata']['image_prompt']=prompt_and_response[0]

        return pdf_dict
    
    @staticmethod
    def _extract_images_from_pdf(path, pdf_image_dir, **kwargs):
        os.makedirs(pdf_image_dir, exist_ok=True)
        images=PDFProcessor._convert_pdf_to_images(path, dpi=kwargs.get('dpi',300))

        for i_page, image in enumerate(images):
            image_path=os.path.join(pdf_image_dir,f'page_{i_page+1}.png')
            PDFProcessor._save_image(image_path,image)

    @staticmethod
    def _extract_information_from_pdf( path):
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)

                pages_info={'metadata':{'pdf_path':path, 'num_pages':num_pages},
                            'pages':{} }
                for page_num in range(num_pages):
                    
                    page = reader.pages[page_num]

                    # Gets text fromt the page
                    page_text=page.extract_text()

                    # Check for images in the page
                    try:
                        images=page.images
                    except:
                        images=[]

                    # This is for when pa file whcih are presentation style pages are present
                    resources=page['/Resources']
                    is_a_pa_attachment=False
                    if '/XObject' in resources:
                        xobjects = page['/Resources']['/XObject']
                        is_a_pa_attachment=True
                    
                    process_as_image=False
                    if (is_a_pa_attachment or len(images)>0 or len(page_text)==0):
                        process_as_image = True

                    pages_info['pages'][f'page_{page_num+1}']={
                                                'text':page_text,
                                                'process_as_image':process_as_image,
                                                }

        except PyPDF2.errors.PdfReadError as e:
            print(f"Error reading PDF: {e}")
            return None
        
        return pages_info
    
    @staticmethod
    def fix_filename(filename):
        # Removing bad characters
        n_dot = filename.count('.')
        if n_dot < 2:
            filename=filename.split('.')[0]
        else:
            filename=filename.replace('.', '',1).split('.')[0]

        filename=filename.replace('%', '_').replace('(', '').replace(')', '').strip()
        return filename
    
    @staticmethod
    def _construct_storage_dict(n_pages):
        pdf_dict={}
        pdf_dict['metadata']={'pdf_name':'',
                              'pdf_path':'',
                              'num_pages':n_pages,
                              'image_prompt':''
                              }
        pdf_dict['pages']={}
        for i_page in range(n_pages):
            pdf_dict['pages'][f'page_{i_page+1}']={'text':'',
                                                   'process_as_image':False}

        return pdf_dict
    
    @staticmethod
    def _save_dict(path, pdf_dict):
        with open(path, 'w') as f:
            json.dump(pdf_dict, f)
        return None
    
    @staticmethod
    def _convert_pdf_to_images(path, dpi=300):
        images = convert_from_path(path, dpi=dpi)
        return images
    
    @staticmethod
    def _save_image(path:str, image:np.ndarray):
        image.save(path, 'PNG')

    


if __name__ == "__main__":



    # pdf_file=os.path.join('data','dft','raw','Thomas_1927.pdf')
    # pdf_file=os.path.join('data','dft','raw','1965-140 PR Kohn & Sham - Self-consistent equations including exchange & correlation effects.pdf')
    # pdf_file=os.path.join('data','dft','raw','Thomas_1927.pdf')
    # processor=PDFProcessor(
    #                     output_dir=os.path.join('data','dft','interim'), 
    #                     model='gpt-4o-mini',
    #                     max_tokens=3000)
    # processor.process(pdf_file, method='llm')

    processor=PDFProcessor(
                        output_dir=os.path.join('data','tiau','interim'), 
                        model='gpt-4o-mini',
                        max_tokens=3000)

    pdf_files=glob(os.path.join('data','tiau','raw','*.pdf'))
    for pdf_file in pdf_files:
        print(pdf_file)
        processor.process(pdf_file, method='llm')

    