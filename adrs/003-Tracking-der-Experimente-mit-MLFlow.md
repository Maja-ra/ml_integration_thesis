---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 003 Tracking der Experimente mit MLFLow

## Kontext and Problem-Statement

Iterative Experimentdurchläufe erzeugen viele Modelle mit verschiedenen Parametern. Welcher Arlgorithmus und welche Parameter sind vergleichsweise am besten? Wie wurde das Modell trainiert? Was wurde getestet? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Ausgereiftheit
* Visualisierungsplattform
* Integrationsmöglichkeiten in andere Komponenten

## Betrachtete Optionen

* Tracking der Experimente mit DVC
* Tracking der Experimente mit MLFlow
* Tracking der Experimente mit Databricks

## Entscheidungsergebnis

Ausgewählte Option: "Tracking der Experimente mit MLFlow", weil es schon ausgereifter, dediziert für Experimente und die Visualisierungsplattform in andere Plattformen integrierbar ist. 

## Beinflussende Empfehlungen

* 7.1
* 7.7
* 7.10
* 7.19
* 7.21
* 8.2

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Skalierbarkeit von Experimentdurchläufen verbessert.
* Gut, weil es die Modell-Optimierung verbessert.
* Gut, weil es die Erklärbarkeit verbessert.
* Gut, weil es die Reproduzierbarkeit verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.
* Schlecht, weil es die Performance Efficiency der Trainingsdurchläufe verschlechtert.