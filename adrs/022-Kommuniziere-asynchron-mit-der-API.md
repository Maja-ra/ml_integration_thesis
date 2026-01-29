---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 022 Kommuniziere asynchron mit der API

## Kontext and Problem-Statement

Eine variierende Anzahl an Nutzern sendet Anfragena an die API. Wird die Anfrage aus Nutzersicht schnell genug verarbeitet? Werden die Ressourcen effizient benutzt? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Ressourcenauslastung
* Antwortzeit
* Komplexität

## Betrachtete Optionen

* Kommuniziere asynchron mit der API
* Kommuniziere synchron mit der API
* Verwende Message‑Queues

## Entscheidungsergebnis

Ausgewählte Option: "Kommuniziere asynchron mit der API", weil Anfragen parallel abgearbeitet werden können ohne dass Ressourcen blockiert werden.  Weiter empfohlene Patterns können auf dieser Basis integriert werden.

## Beinflussende Empfehlungen

* 5.6

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Robustheit verbessert.
* Gut, weil es die Skalierbarkeit verbessert.
* Gut, weil es Performance verbessert.
* Schlecht, weil es die Testbarkeit erschwert.
* Schlecht, weil es das Debugging erschwert.