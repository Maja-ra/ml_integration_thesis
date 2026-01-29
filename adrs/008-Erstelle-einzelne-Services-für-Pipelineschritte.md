---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 008 Erstelle einzelne Services für Pipelineschritte

## Kontext and Problem-Statement

Code-, Daten- und Modell- hängen miteinander zusammen, werden aber getrennt verfolgt. Welche Modelle und Daten werden von dem Code verwendet/erstellt? Wo ist das zugehörige Repository? Kann das zusammen auf einer Plattform angezeigt werden? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Fehlererkennung und Debugging
* Unterstützung von Änderungen
* Ausführung von Teilschritten
* Testbarkeit
* Skalierung

## Betrachtete Optionen

* Erstelle einzelne Services für Pipelineschritte
* Erstelle eine Datei für ETL- und Training-Pipeline

## Entscheidungsergebnis

Ausgewählte Option: "Erstelle einzelne Services für Pipelineschritte", weil besseres Debugging, Änderungsmanagement und unabhängige Ausführung aller Schritte möglich ist.  

## Beinflussende Empfehlungen

* 3.6
* 5.1
* 8.13

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es Skalierbarkeit verbessert.
* Gut, weil es das Debugging verbessert.
* Schlecht, weil es die Orchestrierung aufwendiger macht.