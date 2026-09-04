"""OAP tests for AgenticStringComparator.contains.

Tests the LLM-based contains function with known-positive and
known-negative semantic matching pairs. Each case is its own test.
"""

from __future__ import annotations

import pytest

from peteos.agentic_objects.string_comparator import AgenticStringComparator


@pytest.mark.oap
async def test_contains_positive_1():
    """Test positive match: 'The sky is blue' contains 'The color of the world's ceiling'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains(
        "The sky is blue",
        "The color of the world's ceiling",
    )
    assert result is True, (
        "contains('The sky is blue', 'The color of the world's ceiling')"
    )


@pytest.mark.oap
async def test_contains_positive_2():
    """Test positive match: 'Water boils at 100 degrees Celsius' contains 'Boiling temperatur of water'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains(
        "Water boils at 100 degrees Celsius",
        "Boiling temperatur of water",
    )
    assert result is True, (
        "contains('Water boils at 100 degrees Celsius', 'Boiling temperatur of water')"
    )


@pytest.mark.oap
async def test_contains_positive_3():
    """Test positive match: 'The plant uses sunlight to produce energy' contains 'energy production'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains(
        "The plant uses sunlight to produce energy",
        "energy production",
    )
    assert result is True, (
        "contains('The plant uses sunlight to produce energy', 'energy production')"
    )


@pytest.mark.oap
async def test_contains_negative_1():
    """Test negative match: 'The sky is blue' does not contain 'green'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains("The sky is blue", "green")
    assert result is False, (
        "contains('The sky is blue', 'green') should be False"
    )


@pytest.mark.oap
async def test_contains_negative_2():
    """Test negative match: 'Water boils at 100 degrees Celsius' does not contain 'freezing temperature'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains(
        "Water boils at 100 degrees Celsius",
        "freezing temperature",
    )
    assert result is False, (
        "contains('Water boils at 100 degrees Celsius', 'freezing temperature')"
    )


@pytest.mark.oap
async def test_contains_negative_3():
    """Test negative match: 'Photosynthesis converts sunlight into energy' does not contain 'nuclear energy'."""
    comp = AgenticStringComparator.instance()
    result = await comp.contains(
        "Photosynthesis converts sunlight into energy",
        "nuclear energy",
    )
    assert result is False, (
        "contains('Photosynthesis converts sunlight into energy', 'nuclear energy')"
    )
