---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 027 Führe einen Deployment-Check durch

## Kontext and Problem-Statement

Die Performance des Modells kann bei Retraining beeinflusst werden. Ist diese ausreichend? Ist ein Re-Deployment notwendig? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Sicherstellung Modell-Performance
* Deployment-Intervall

## Betrachtete Optionen

* Führe einen Deployment-Check durch
* Führe nach jedem Retraining Deployment-Check durch

## Entscheidungsergebnis

Ausgewählte Option: "Führe einen Deployment-Check durch", weil dadurch die Qualität des Modells sichergestellt und die Anzahl der Deployments auf das Notwendigste begrenzt wird.  

## Beinflussende Empfehlungen

* 13.8

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Modellgüte verbessert.
* Gut, weil es die Robustheit verbessert.
* Gut, weil es die Verfügbarkeit verbessert.