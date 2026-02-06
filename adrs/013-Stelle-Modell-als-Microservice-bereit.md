---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 013 Stelle Modell als Microservice bereit

## Kontext and Problem-Statement

Das entwickelte Modell muss mit der restlichen Software Verknupft werden. Wie soll dieser Zusammenschluss aussehen? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Zuverlässigkeit des Systems
* Vorhandene Softwarestruktur
* Programmiersprachen
* Skalierbarkeit
* Wiederverwendbarkeit und Modularität

## Betrachtete Optionen

* Stelle Modell als Mircroservice bereit
* Stelle Software als Monolith bereit
* Erstelle adaptive Software mit selbstlernenden Komponenten

## Entscheidungsergebnis

Ausgewählte Option: "Stelle Modell als Mircroservice bereit", weil es dadurch separat gemanaged und skaliert werden kann sowie unabhängig/mit weniger Abhängigkeiten entwickelt und deployed werden kann. 

## Beinflussende Empfehlungen

* 4.4
* 5.4

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Skalierbarkeit verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Wiederverwendbarkeit verbessert.
* Gut, weil es die Änderbarkeit verbessert.
* Schlecht, weil es den Netzwerk-Overhead erhöht.
* Schlecht, weil es die Komplexität erhöht.
* Schlecht. weil robustes Monitoring notwendig ist.
* Schlecht, weil es die Sicherheit verschlechtert.