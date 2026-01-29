---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 025 Visualisiere Klassifikation mit Shap

## Kontext and Problem-Statement

Die Klassifikation wird von dem Alogithmus bestimmt. Basierend auf welchen Merkmalen werden die Entscheidungen getroffen? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Sicherstellung Modell-Performance
* Deploymentintervall

## Betrachtete Optionen

* Visualisiere Klassifikation mit Shap
* Visualisiere Klassifikation mit Lime
* Visualisiere Klassifikation nicht

## Entscheidungsergebnis

Ausgewählte Option: "Visualisiere Klassifikation mit Shap", weil die Ergebnisse für alle Stakeholder transparenter sind.  

## Beinflussende Empfehlungen

* 7.4
* 11.7
* 13.4
* 13.5

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Erklärbarkeit verbessert.