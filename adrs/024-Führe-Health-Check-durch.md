---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 024 Führe Health Check durch

## Kontext and Problem-Statement

Frontend ist von Status und Antwort des Services abhängig. Ist der Service funktionsfähig? Kann der der Service aufgerufen werden? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* User Experience
* Fehleranfälligkeit

## Betrachtete Optionen

* Führe Health Check durch
* Führe keinen Health Check durch

## Entscheidungsergebnis

Ausgewählte Option: "Führe Health Check durch", weil basierend auf dem Ergebnis das System entsprechend reagieren kann und zuverlässig agieren kann.  

## Beinflussende Empfehlungen

* 

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Usability verbessert.
* Gut, weil es die Robustheit verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.