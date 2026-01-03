---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 020 Führe Threshhold für Modellperformance ein

## Kontext and Problem-Statement

Mit neuen Daten kann das Modell neu trainiert werden. Ist die Performance des Modells genügend? Ist ein Deployment notwendig? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Sicherstellung Modell-Performance
* Deploymentintervall

## Betrachtete Optionen

* Führe Threshhold für Modellperformance ein
* Deploye keine neuen Modellversionen
* Deploye Modellversionen bei jeden neuen Daten

## Entscheidungsergebnis

Ausgewählte Option: "Führe Threshhold für Modellperformance ein", weil die Anzahl an Deployments verringert und eine bestimmtes Performance-Level des Modells gewährleistet wird.  

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Performance verbessert.
* Gut, weil es die Availability verbessert.
* Gut, weil es Ressourceneffizienz verbessert.