from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import GenerationConfig
from typing_extensions import TypedDict
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

trained_model_dir = r"D:\university courses\GenAI\asighnments\Assighnment-4\partB\agents\trained_model"

trained_tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)
trained_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_dir)

class State(TypedDict):
    original_word: str
    key_words:None
    article_texts:None
    summaries:None
    comparison_results:None

def save_pdf_report(topic_summary,papers):
    """
    Generate a structured research report PDF.

    Args:
        topic_summary (str): Brief overview of the topic.
        papers (list): List of paper dicts with keys: title, authors, summary, methodology, contributions, limitations.
        comparative_analysis (dict): Dict with keys: common_findings, conflicts, gaps.
        file_path (str): File path to save the PDF.
    """
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Research Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 24))

    # Topic Summary
    elements.append(Paragraph("Topic Summary", styles['Heading2']))
    elements.append(Paragraph(topic_summary, styles['Normal']))
    elements.append(Spacer(1, 18))

    # Top Papers
    elements.append(Paragraph("Top Papers List", styles['Heading2']))
    for idx, paper in enumerate(papers, 1):
        elements.append(Paragraph(f"{idx}. {paper[4]}", styles['Heading3']))
        elements.append(Paragraph(f"Authors: {paper[5]}", styles['Normal']))
        elements.append(Paragraph("Full Summary:", styles['Heading4']))
        elements.append(Paragraph(paper[1], styles['Normal']))
        elements.append(Paragraph("journal Ref:", styles['Heading4']))
        elements.append(Paragraph(paper[2], styles['Normal']))
        elements.append(Paragraph("Publish Date:", styles['Heading4']))
        elements.append(Paragraph(paper[3], styles['Normal']))
        elements.append(Spacer(1, 18))

    # Build PDF
    doc.build(elements)
    file_path="research_report.pdf"
    print(f"PDF saved to: {file_path}")

def paper_summary_generator(state:State):
    papers=state['article_texts']
    summaries=[]

    for paper in papers:
        paper=paper[0]

        prompt = f"""
        Summarize the following conversation.
        {paper}
        Summary:
        """

        trained_input_ids = trained_tokenizer(prompt, return_tensors="pt").input_ids
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        trained_input_ids = trained_input_ids.to(device)
        
        trained_model.to(device)
        
        # Generate outputs using the original model before training
        generation_config = GenerationConfig(max_new_tokens=200, num_beams=1)
        
        # Generate outputs using the trained model
        trained_model_outputs = trained_model.generate(input_ids=trained_input_ids, generation_config=generation_config)
        trained_model_text_output = trained_tokenizer.decode(trained_model_outputs[0], skip_special_tokens=True)

        summaries.append(trained_model_text_output)
    
    state['summaries']=summaries
    
    save_pdf_report(summaries[0],papers)

    return state