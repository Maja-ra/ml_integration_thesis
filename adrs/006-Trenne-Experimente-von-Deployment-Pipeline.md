---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 006 Trenne Experimente von Deployment Pipeline

## Kontext and Problem-Statement

Experimente werden anders gemanaged als das Modell in der Entwicklung. Werden Experimente beim Deployment benötigt? Wie beeinflussen Experimente nicht das Deployment? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Unabhängigkeit der Experimentdurchläufe

## Betrachtete Optionen

* Trenne Experimente von Deployment Pipeline
* Füge Experimente als Pipeline-Schritt hinzu

## Entscheidungsergebnis

Ausgewählte Option: "Trenne Experimente von Deployment Pipeline", weil dadurch unabhängig vom Deployment experimentiert werden kann und weniger weniger Deployment-Risiken entstehen.

## Beinflussende Empfehlungen

* 5.2

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Schlecht, weil es duplizierte Codeanteile erhöht.