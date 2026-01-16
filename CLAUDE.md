# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

structure-search is an early-stage Python project for protein structure search and analysis. It will utilize AlphaFold protein structure databases via Foldseek.

## Current State

This repository is in initial setup. No source code, build system, or tests exist yet. The `data/` directory contains a symlink to AlphaFold Swiss-Prot structure databases (Foldseek format).

## Recommended Setup

When establishing the development infrastructure:

- **Package management**: uv or Poetry
- **Testing**: pytest
- **Linting/formatting**: Ruff
- **Type checking**: mypy

## Data Directory

The `data/foldseek/` symlink points to AlphaFold Swiss-Prot protein structure databases in Foldseek format, including:
- Main structure data (`alphafold_swissprot`)
- C-alpha atoms (`alphafold_swissprot_ca`)
- Secondary structure (`alphafold_swissprot_ss`)
- Associated index and lookup files
