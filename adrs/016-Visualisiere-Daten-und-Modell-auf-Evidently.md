---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 016 Visualisiere Daten und Modelle auf Evidently

## Kontext and Problem-Statement

Die Daten sind in ihrer Gesamtheit ohne Visualisierungen schwierig zu verstehen. Wie werden die Daten visualisiert und wo werden die Visualisierungen angezeigt? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Austausch von Visualisierungen mit dem Team
* Zeitpunkt der Einführung von Visualisierungen
* Funktionsumfang Daten, Modell und Vorhersagen
* Zusammenhängende Dokumentation

## Betrachtete Optionen

* Visualisiere Daten und Modelle auf Evidently
* Visualisiere Daten und Modelle mit Kibana
* Visualisiere Daten und Modelle nur beim Proframmieren/bei der EDA

## Entscheidungsergebnis

Ausgewählte Option: "Visualisiere Daten und Modelle auf Evidently", weil Reports zu wichtigen Daten- und Modelleigenschaften automatisch visualisiert, Alerts generiert werden können und Vorhersagen als Datensatz angelegt werden können. Außerdem werden Visualisierungen schon vor dem Deployment individuell für den MLOps-Workflow benötigt. Die Plattform ist speziell für ML-Anwendungsfälle.

## Beinflussende Empfehlungen

* 3.3
* 4.10
* 7.11
* 7.13
* 7.15
* 8.11
* 11.3

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wartbarkeit verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.
* Gut, weil es die Rückverfolgbarkeit verbessert.
* Schlecht, weil es eine weitere technologische Abhängigkeit hinzufügt.