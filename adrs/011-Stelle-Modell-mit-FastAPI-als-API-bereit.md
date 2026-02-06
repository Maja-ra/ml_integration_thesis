---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 011 Stelle Modell mit FastAPI als API zur Verfügung

## Kontext and Problem-Statement

Die Open-Source-Software benutzt das ML-Modell um Vorhersagen zu treffen. Wie ist das Modell zugänglich für die Software? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Skalierbarkeit
* Performance
* Anpassungsmöglichkeiten und iteratives Deployment

## Betrachtete Optionen

* Stelle Modell mit FastAPI als API zur Verfügung
* Stelle Modell mit Flask als API zur Verfügung
* Integriere ML-Modell direkt in den Code

## Entscheidungsergebnis

Ausgewählte Option: "Stelle Modell mit FastAPI als API zur Verfügung", weil alle erfassten Mertadaten zusammen dargestellt sowie schnell und übersichtlich einsehbar sind. Ermöglicht die Integration von DVC und MLFlow. 

## Beinflussende Empfehlungen

* 9.8
* 10.5

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Skalierbarkeit verbessert.
* Gut, weil es die Wiederverwendbarkeit verbessert.
* Gut, weil es die Performance verbessert.
* Gut, weil es die Änderbarkeit verbessert.
* Gut, weil es die Interoperabilität verbessert.
* Schlecht, weil es die Entwicklungszeit verlängert.
* Schlecht, weil es den Netzwerk-Overhead erhöht.