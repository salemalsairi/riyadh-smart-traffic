from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(file, CS, CO):

    with open(file,'r', encoding='utf-8') as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= int(CS),
        chunk_overlap= int(CO)

    )
    output = splitter.split_text(text)
    return output
kas_text=split_text('kapsarc.txt', 1200, 120)

school_text = split_text('schools_tti.txt', 800, 120)

if __name__ == "__main__":
    print(len(kas_text))
    print(kas_text[0])
if __name__ == "__main__":
    print(len(school_text))
    print(school_text[0])
