"""Exportações e anotação independentes da interface web."""
import base64
import io
import json
from PIL import Image, ImageDraw, ImageFont
from .nomes import display_name, display_report


def preview_image(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')


def annotated_image(result):
    image = preview_image(result['image'])
    draw = ImageDraw.Draw(image)
    size = max(13, round(image.width / 55))
    font = ImageFont.load_default(size=size)
    for face in result['faces']:
        box = face['box']
        color = '#218568' if face['accepted'] else '#ad731d'
        draw.rectangle(box, outline=color, width=max(2, round(image.width / 350)))
        text = f"{face['id']} · {display_name(face['label'])}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width, height = right-left+12, bottom-top+10
        x = max(0, min(box[0], image.width-width))
        y = max(0, box[1]-height)
        draw.rectangle((x, y, x+width, y+height), fill=color)
        draw.text((x+6, y+5-top), text, fill='white', font=font)
    return image


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return buffer.getvalue()


def report_json(result):
    return json.dumps(display_report(result['report']), ensure_ascii=False, indent=2)


def report_text(result):
    report = display_report(result['report'])
    lines = ['AP02 · Diagnóstico de reconhecimento', report['purpose'],
             f"Arquivo: {report['source']['filename']}", f"Data: {report['analyzed_at']}",
             f"SHA256: {report['source']['sha256']}",
             f"Modelo: {(report['model'] or {}).get('run_id', 'não treinado')}"]
    for row in report['faces']:
        score = 'sem modelo' if row['score'] is None else f"{row['score']:.2%}"
        lines.append(f"Rosto {row['id']}: {row['label']} | pontuação {score} | limiar {row['threshold']}")
    return '\n'.join(lines)
