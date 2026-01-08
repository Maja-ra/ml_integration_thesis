---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 004 Führe Versionierungen auf Dagshub zusammen

## Kontext and Problem-Statement

Code-, Daten- und Modell- hängen miteinander zusammen, werden aber getrennt verfolgt. Welche Modelle und Daten werden von dem Code verwendet/erstellt? Wo ist das zugehörige Repository? Kann das zusammen auf einer Plattform angezeigt werden? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Unübersichtliche, verteilte Metadaten-Sammlungen
* Zusammenhängende Dokumentation

## Betrachtete Optionen

* Führe Versionierungen auf Dagshub zusammen

## Entscheidungsergebnis

Ausgewählte Option: "Führe Versionierungen auf Dagshub zusammen", weil alle erfassten Mertadaten zusammen dargestellt sowie schnell und übersichtlich einsehbar sind. Ermöglicht die Integration von DVC und MLFlow. 

## Beinflussende Empfehlungen

* 3.14
* 11.11

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.