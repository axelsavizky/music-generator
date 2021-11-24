# Music generator

## Datasets:
- Schubert short dataset: https://drive.google.com/file/d/1qnQVK17DNVkU19MgVA4Vg88zRDvwCRXw/view
- Gwern long dataset: https://goo.gl/VezNNA

## Como correr perplexity
python feedforward.py model\_path dict\_path TEST\_SET\_SIZE

## Como generar canción
python generate_song.py model\_path dict\_path sample\_song\_path beam\_search\_k generated\_song\_name \[max\_consecutive\_note\_repetition\] \[tabu\_list\_length\]
