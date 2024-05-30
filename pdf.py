import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("bluiding index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return  index

pdf_path = os.path.join('data', 'brazil.pdf')
brazil_pdf = PDFReader().load_data(file=pdf_path)
brazil_index = get_index(brazil_pdf, 'Brazil')
brazil_engine = brazil_index.as_query_engine()
