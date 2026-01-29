---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 023 Führe Integrationstests mit pytest durch

## Kontext and Problem-Statement

Mehrere Komponenten werden verbunden, die korrekt zusammenarbeiten müssen. Sind die Komponenten korrekt verbunden? Werden die richtigen Daten übertragen? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Risikominimierung
* Aufwand

## Betrachtete Optionen

* Führe Integrationstests mit pytest durch
* Führe Unittests durch
* Führe Contract-Tests durch
* End‑to‑End‑Tests durch

## Entscheidungsergebnis

Ausgewählte Option: "Führe Integrationstests mit pytest durch", weil die Anzahl an Deployments verringert und eine bestimmtes Performance-Level des Modells gewährleistet wird.  

## Beinflussende Empfehlungen

* 12.2
* 12.4

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Robustheit verbessert.
* Gut, weil es die Verfügbarkeit verbessert.
* Schlecht, weil nicht das Gesamtsystem überprüft wird.