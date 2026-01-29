---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 030 Erstelle einen Klassifizierungsbericht zur Bewertung des ML-Modells

## Kontext and Problem-Statement

Daten verändern sich, wodurch ein Drift entstehen kann. Ist ein Retraining notwendig? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Bewertung der Modell-Leistung
* Visualisierbarkeit
* Vergleichbarkeit

## Betrachtete Optionen

* Erstelle einen Klassifizierungsbericht zur Bewertung des ML-Modells
* Bewerte Modell mit Accuracy-Score

## Entscheidungsergebnis

Ausgewählte Option: "Erstelle einen Klassifizierungsbericht zur Bewertung des ML-Modells", weil es eine besserer Bewertungsgrundlage für die Leistung des Modells basierend auf mehreren Metriken bildet und verglichen werden kann. 

## Beinflussende Empfehlungen

* 7.11
* 7.13
* 7.15
* 11.5
* 11.8

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Modellgüte verbessert.
* Gut, weil es die Robustheit verbessert.
* Schlecht, weil es die Erklärbarkeit erhöht.
* Schlecht, weil es das Vertrauenswürdigkeit erhöht.