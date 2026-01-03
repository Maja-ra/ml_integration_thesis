---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 012 Containerisiere mit Docker

## Kontext and Problem-Statement

...

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Plattformunabhängigkeit
* Kosistenz
* Ressourceneffizienz
* Bereitstellungsgeschwindigkeit
* CI/CD

## Betrachtete Optionen

* Containerisiere mit Docker
* Containerisiere mit Podman
* Verwende eine VM
* Keine Containerisierung

## Entscheidungsergebnis

Ausgewählte Option: "Containerisiere mit Docker", weil eine konsistente, plattformunabhängige, modulare Umgebung erschaffen wird sowie sich bessere Möglichkeiten zur Orchestrierung und für CI/CD ergeben.

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Skalierbarkeit verbessert.
* Gut, weil es die Reproduzierbarkeit verbessert.
* Gut, weil es die Ressourceneffizienz verbessert.
* Schlecht, weil sich dich Komplexität der Orchestrierung erhöht.
* Schlecht, weil neue Sicherheitslücken entstehen.
* Schlecht, weil die Perfermance sich verschlechtert.
* Neutral, weil Isolierung geringer als bei vollständigen Betriebssystemen.
