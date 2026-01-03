---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 007 Benutze Gradient-Boost-Klassifizierer

## Kontext and Problem-Statement

Mehrer Algorithmen können Vorhersage durchführen. Welcher Alorithmus soll verwendet werden?  

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Modell-Performance
* Erklärbarkeit
* Benötigte Rechenleistung

## Betrachtete Optionen

* Benutze Gradient-Boost-Klassifizierer
* Benutze Decision-Tree-Klassifizierer
* Benutze Logistiche Regression
* Benutze Random-Forrest-Klassifizierer
* Benutze Neuronales Netz

## Entscheidungsergebnis

Ausgewählte Option: "Benutze Gradient-Boost-Klassifizierer", weil der Klassifizierer die beste Modell-Performance liefert und Erklärbarkeit ermöglicht. 

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Erklärbarkeit verbessert.
* Gut, weil es die Performance Efficiency verbessert.
* Schlecht, weil Erklärbarkeit einzelner Features gespalten sind.