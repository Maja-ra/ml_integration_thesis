---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 021 Lagere Parameter aus und passe dynamisch an

## Kontext and Problem-Statement

Während der Durchführung der Pipeline werden verschiedene Parameter zu Konfiguration verwendet. Wo werden dieses Parameter derfiniert? Wie werden Änderungen übernommen? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Sicherstellung Modell-Performance
* Deploymentintervall

## Betrachtete Optionen

* Lagere Parameter aus und passe dynamisch an
* Definiere Paramter im Service
* Lagere Parameter aus

## Entscheidungsergebnis

Ausgewählte Option: "Lagere Parameter aus und passe dynamisch an", weil Parameter durchgehend verwendet werden und deren Konsistenz sowie unkomplizierte Änderungsmöglichkeiten sichergestellt werden.  

## Beinflussende Empfehlungen

* 4.6
* 3.8
* 3.6
* 3.10
* 4.15
* 4.18
* 7.17
* 7.18

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Änderbarkeit verbessert.
* Gut, weil es die Datenintegrität verbessert.
* Gut, weil es Nachvollziehbarkeit verbessert.
* Gut, weil es die Anpassungsfähigkeit verbessert.
* Schlecht, weil es die Komplexität erhöht.