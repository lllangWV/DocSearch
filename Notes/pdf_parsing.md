# PDF Parsing

### Determining if a pdf page should be processed as an image or text

```python
def extract_information_from_pdf(pdf_file):

    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            pages={}
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
                # Look at data/minutes/raw/august-22-2023-bog-minutes-with-attachments.pdf'
                resources=page['/Resources']
                is_a_pa_attachment=False
                if '/XObject' in resources:
                    xobjects = page['/Resources']['/XObject']
                    is_a_pa_attachment=True
                
                process_as_image=False
                if (is_a_pa_attachment or len(images)>0 or len(page_text)==0):
                    process_as_image = True

                pages[f'page_{page_num+1}']={
                                            'text':page_text,
                                            'process_as_image':process_as_image,
                                            # 'is_a_pa_attachment':is_a_pa_attachment
                                            }

    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF: {e}")
        return None
    
    return pages
```


In the case of november 17, 2023 bog meeting minutes page 17, text was extracted from the orginal pdf file, but it was still processed as an image.

This is due to `(is_a_pa_attachment or len(images)>0 or len(page_text)==0)` signal true. It probably had both text and image and was processed as an image to be safe.