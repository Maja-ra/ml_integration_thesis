---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 017 Definiere Schnittstellen/Data Contracts für API und ML-Schritte

## Kontext and Problem-Statement

Die verschiedenen Services geben Daten weiter und erhalten Daten mit denen sie arbeiten. Welche Daten soll die Anfrage enthalten? In welchen Format? Welche Daten werden erstellt und wie abgespeichert? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Datenkonsistenz
* Standardisierung
* Automatisierung

## Betrachtete Optionen

* Definiere Schnittstellen-Data Contracts für API und ML-Schritte

## Entscheidungsergebnis

Ausgewählte Option: "Definiere Schnittstellen-Data Contracts für API", weil Daten konsistent und automatisch verarbeitet werden können sowie Änderungen in einem Modul leichter möglich sind.  

## Beinflussende Empfehlungen

* 3.11
* 4.17
* 13.8

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es das Debugging verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Sicherheit verbessert.
* Gut, weil es die Änderbarkeit verbessert.
* Gut, weil es die Testbarkeit verbessert.
* Neutral, weil Fehlermeldungen durch Abweichungen entstehen.