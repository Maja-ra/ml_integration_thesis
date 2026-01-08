---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 009 Automatisiere Pipeline mit DVC

## Kontext and Problem-Statement

Prozesschritte bauen aufeinander auf und müssen in betsimmter Reihenfolge durchgeführt werden. Was ist die Reihenfolge und was sind die Abhängigkeiten der Prozesschritte? Müssen alle Schritte durchgeführt werden? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Redundanz von Pipeline-Phasen bei mehreren Durchläufen
* Manuelle Orchestrierung einzelner Schritte
* Reproduzierbarkeit und Wiederholbarkeit

## Betrachtete Optionen

* Automatisiere Pipeline mit DVC
* Automatisiere Pipeline mit Jenkins
* Automatisiere Pipeline mit GitHub Actions

## Entscheidungsergebnis

Ausgewählte Option: "Automatisiere Pipeline mit DVC", weil Abhängigkeiten erfasst werden, Fehler gemeldet werden, nur notwendige Phasen ausgeführt werden und der Ablauf wiederhoöbar definiert ist. 

## Beinflussende Empfehlungen

* 3.12
* 4.8
* 4.13
* 8.7
* 9.1

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Performance Efficiency verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.
* Gut, weil es die Reproduzierbarkeit verbessert.
* Schlecht, weil der Umfang der Funktionen eingeschränkt ist.