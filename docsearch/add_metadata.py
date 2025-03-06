import json
import os
import shutil
from glob import glob

from image_processing import IMAGE_EXTRACTION_PROMPT
pdf_dir_path=os.path.join('data','minutes','interim')


pdf_dirs=glob(os.path.join(pdf_dir_path,'*'))
# for pdf_dir in pdf_dirs:
#     print(pdf_dir)
#     combined_pdf_file=os.path.join(pdf_dir,f'combined_pdf_info.json')
#     old_processed_file=os.path.join(pdf_dir,f'pdf_info.json')
#     new_processed_file=os.path.join(pdf_dir,f'preproceed_pdf_info.json')
#     try:
#         os.rename(old_processed_file, new_processed_file)
#         shutil.rmtree(old_processed_file)
#     except:
#         pass

#     with open(combined_pdf_file, 'r') as f:
#         data = json.load(f)

#     pdf_filename=os.path.basename(pdf_dir)

#     pdf_path=os.path.join('data','minutes','raw',pdf_filename,'.pdf')
    
#     new_data={}
#     new_data['metadata']={
#         'pdf_name':pdf_filename,
#         'pdf_path':pdf_path,
#         'processed_path':pdf_dir,
#         'num_pages':len(data),
#         'image_prompt':IMAGE_EXTRACTION_PROMPT
#     }
#     new_data['pages']=data

#     with open(old_processed_file, 'w') as f:
#         json.dump(new_data, f)

for pdf_dir in pdf_dirs:
    print(pdf_dir)
    pdf_file=os.path.join(pdf_dir,f'pdf_info.json')

    with open(pdf_file, 'r') as f:
        data = json.load(f)
    new_pages_dict={}
    for key, page_dict in data.get('pages',{}).items():
        new_pages_dict[key]={}
        new_pages_dict[key]['text']=page_dict

    new_data={
        'metadata':data.get('metadata',{}),
        'pages':new_pages_dict
    }

    with open(pdf_file, 'w') as f:
        json.dump(new_data, f)