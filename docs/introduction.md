# Introduction

Peteos is a library for developing agentic software, built on a paradigm called **Simple Object-Agentic Programming (sOAP)** — a paradigm that marries object-oriented programming with agentic systems.

Traditionally, object-oriented programming gives you objects with well-defined state and behavior. Agentic systems give you autonomous agents that perceive, reason, and act. Peteos brings these two worlds together, letting you build objects that are truly active — objects that can think, decide, and interact with their environment on their own.

## What Problem Does This Solve?

Consider an NPC class in a game. The NPC needs to decide what to do based on its current situation, goals, and surroundings. Consider a robot that navigates an environment and must determine when to move, when to avoid obstacles, and when to complete a task. Consider data cleansing — given a table of unstructured rows, putting that data into a structured format is one of the hardest challenges, especially when the incoming data structure is entirely unknown.

Modern AI technology, including LLMs and agentic systems, enables solutions to these kinds of problems. But building complex software that naturally weaves this technology into its core is difficult. Most agentic frameworks treat agents as external actors — separate from your domain objects. This creates a disconnect between your business logic and the intelligence that drives it.

## The Agentic Object

This is where Peteos closes the gap. It introduces the concept of **agentic objects** — objects that can be invoked, reasoned with, and empowered to interact with their own state.

An agentic object is an object of an agentic class. It can be invoked or "spoken to," and it has the ability to observe and modify its own internal state. It perceives its environment, makes decisions, and takes actions — all while remaining a first-class object within your OOP codebase.

## A Shift in Persistence

Conventional agentic systems rely on memory stores, databases, or external storage to persist information across sessions. In Peteos, the object itself is the center of persistence. The agentic object maintains its own state, its own memory, and its own agency. This means intelligence is not bolted on as an afterthought — it lives at the heart of your domain model.
