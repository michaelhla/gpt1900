#!/usr/bin/env python3
"""
Programmatic collection of canonical pre-1900 physics texts.

Sources:
  - Wikisource (MediaWiki Parse API → HTML → plaintext)
  - Project Gutenberg (direct UTF-8 text download)
  - Internet Archive (OCR djvu.txt files)

Usage:
  python scripts/pre1900_scripts/collect_physics_books.py --dry-run
  python scripts/pre1900_scripts/collect_physics_books.py --only newton_opticks.txt
  python scripts/pre1900_scripts/collect_physics_books.py --source wikisource
  python scripts/pre1900_scripts/collect_physics_books.py  # download all
"""

import argparse
import html
import logging
import re
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated work list
# ---------------------------------------------------------------------------
# Each entry: {
#   "filename": output filename,
#   "source": "wikisource" | "gutenberg" | "ia",
#   "id": source-specific identifier,
#   "author": str, "year": int/str, "title": str,
#   "wikisource_type": "page" | "subpages" | "prefix",  (wikisource only)
#   "prefix": str,  (for subpage-based works)
# }

CURATED_WORKS = [
    # ======================================================================
    # TIER 1: Core Canonical Texts
    # ======================================================================
    {
        "filename": "newton_opticks.txt",
        "source": "gutenberg",
        "id": 33504,
        "author": "Isaac Newton",
        "year": 1704,
        "title": "Opticks",
    },
    {
        "filename": "galileo_two_new_sciences.txt",
        "source": "wikisource",
        "id": "Dialogues_Concerning_Two_New_Sciences",
        "wikisource_type": "subpages",
        "author": "Galileo Galilei",
        "year": 1638,
        "title": "Dialogues Concerning Two New Sciences",
    },
    {
        "filename": "maxwell_treatise_em_full.txt",
        "source": "wikisource",
        "id": "A_Treatise_on_Electricity_and_Magnetism",
        "wikisource_type": "subpages",
        "author": "James Clerk Maxwell",
        "year": 1873,
        "title": "A Treatise on Electricity and Magnetism",
    },
    {
        "filename": "maxwell_theory_of_heat.txt",
        "source": "ia",
        "id": "theoryheat03maxwgoog",
        "author": "James Clerk Maxwell",
        "year": 1872,
        "title": "Theory of Heat",
    },
    {
        "filename": "faraday_experimental_researches_electricity.txt",
        "source": "gutenberg",
        "id": 14986,
        "author": "Michael Faraday",
        "year": 1839,
        "title": "Experimental Researches in Electricity, Vol 1",
    },
    {
        "filename": "carnot_motive_power_of_heat.txt",
        "source": "wikisource",
        "id": "Reflections_on_the_Motive_Power_of_Heat",
        "wikisource_type": "page",
        "author": "Sadi Carnot",
        "year": 1824,
        "title": "Reflections on the Motive Power of Heat",
    },
    {
        "filename": "gibbs_scientific_papers_vol1.txt",
        "source": "wikisource",
        "id": "Scientific_Papers_of_Josiah_Willard_Gibbs,_Volume_1",
        "wikisource_type": "subpages",
        "author": "J. Willard Gibbs",
        "year": 1870,
        "title": "Scientific Papers of J.W. Gibbs, Volume 1",
    },
    {
        "filename": "gibbs_scientific_papers_vol2.txt",
        "source": "wikisource",
        "id": "Scientific_Papers_of_Josiah_Willard_Gibbs,_Volume_2",
        "wikisource_type": "subpages",
        "author": "J. Willard Gibbs",
        "year": 1880,
        "title": "Scientific Papers of J.W. Gibbs, Volume 2",
    },
    {
        "filename": "rayleigh_theory_of_sound.txt",
        "source": "ia",
        "id": "theorysound08raylgoog",
        "author": "Lord Rayleigh",
        "year": 1878,
        "title": "The Theory of Sound",
    },
    {
        "filename": "hertz_electric_waves.txt",
        "source": "ia",
        "id": "electricwavesbe00hertgoog",
        "author": "Heinrich Hertz",
        "year": 1893,
        "title": "Electric Waves",
    },
    {
        "filename": "thomson_tait_natural_philosophy.txt",
        "source": "ia",
        "id": "TreatiseOnNaturalPhilosophyI",
        "author": "Thomson & Tait",
        "year": 1867,
        "title": "Treatise on Natural Philosophy, Vol 1",
    },
    {
        "filename": "clausius_mechanical_theory_of_heat.txt",
        "source": "ia",
        "id": "mechanicaltheor04claugoog",
        "author": "Rudolf Clausius",
        "year": 1867,
        "title": "The Mechanical Theory of Heat",
    },
    {
        "filename": "tyndall_heat_mode_of_motion.txt",
        "source": "ia",
        "id": "heatconsidereda07tyndgoog",
        "author": "John Tyndall",
        "year": 1865,
        "title": "Heat Considered as a Mode of Motion",
    },
    {
        "filename": "tyndall_sound.txt",
        "source": "gutenberg",
        "id": 54969,
        "author": "John Tyndall",
        "year": 1875,
        "title": "Sound",
    },
    {
        "filename": "helmholtz_sensations_of_tone.txt",
        "source": "ia",
        "id": "onsensationston00helmgoog",
        "author": "Hermann von Helmholtz",
        "year": 1875,
        "title": "On the Sensations of Tone",
    },
    {
        "filename": "helmholtz_conservation_of_force.txt",
        "source": "wikisource",
        "id": "On_the_Conservation_of_Force",
        "wikisource_type": "page",
        "author": "Hermann von Helmholtz",
        "year": 1862,
        "title": "On the Conservation of Force",
    },
    {
        "filename": "franklin_electricity.txt",
        "source": "wikisource",
        "id": "Experiments_and_Observations_on_Electricity",
        "wikisource_type": "page",
        "author": "Benjamin Franklin",
        "year": 1751,
        "title": "Experiments and Observations on Electricity",
    },
    {
        "filename": "gilbert_on_the_magnet.txt",
        "source": "wikisource",
        "id": "On_the_Magnet",
        "wikisource_type": "subpages",
        "author": "William Gilbert",
        "year": 1600,
        "title": "On the Magnet",
    },
    {
        "filename": "laplace_system_of_the_world.txt",
        "source": "ia",
        "id": "systemworld02laplgoog",
        "author": "Pierre-Simon Laplace",
        "year": 1830,
        "title": "The System of the World",
    },
    {
        "filename": "laplace_essay_on_probabilities.txt",
        "source": "gutenberg",
        "id": 58881,
        "author": "Pierre-Simon Laplace",
        "year": 1814,
        "title": "A Philosophical Essay on Probabilities",
    },
    # ======================================================================
    # TIER 2: Important Works & Papers
    # ======================================================================
    {
        "filename": "galileo_natation_of_bodies.txt",
        "source": "wikisource",
        "id": "Discourse_Concerning_the_Natation_of_Bodies",
        "wikisource_type": "page",
        "author": "Galileo Galilei",
        "year": 1663,
        "title": "Discourse Concerning the Natation of Bodies",
    },
    {
        "filename": "galileo_sidereal_messenger.txt",
        "source": "gutenberg",
        "id": 46036,
        "author": "Galileo Galilei",
        "year": 1610,
        "title": "The Sidereal Messenger",
    },
    {
        "filename": "faraday_experimental_researches_chemistry.txt",
        "source": "wikisource",
        "id": "Experimental_Researches_in_Chemistry_and_Physics",
        "wikisource_type": "subpages",
        "author": "Michael Faraday",
        "year": 1859,
        "title": "Experimental Researches in Chemistry and Physics",
    },
    {
        "filename": "faraday_forces_of_matter.txt",
        "source": "wikisource",
        "id": "The_Forces_of_Matter",
        "wikisource_type": "subpages",
        "author": "Michael Faraday",
        "year": 1860,
        "title": "The Forces of Matter",
    },
    {
        "filename": "faraday_chemical_history_candle.txt",
        "source": "wikisource",
        "id": "The_Chemical_History_of_a_Candle",
        "wikisource_type": "subpages",
        "author": "Michael Faraday",
        "year": 1861,
        "title": "The Chemical History of a Candle",
    },
    {
        "filename": "faraday_various_forces_of_nature.txt",
        "source": "wikisource",
        "id": "On_the_various_forces_of_nature_and_their_relations_to_each_other",
        "wikisource_type": "subpages",
        "author": "Michael Faraday",
        "year": 1863,
        "title": "On the various forces of nature",
    },
    {
        "filename": "maxwell_faradays_lines_of_force.txt",
        "source": "wikisource",
        "id": "On_Faraday's_Lines_of_Force",
        "wikisource_type": "page",
        "author": "James Clerk Maxwell",
        "year": 1856,
        "title": "On Faraday's Lines of Force",
    },
    {
        "filename": "maxwell_physical_lines_of_force.txt",
        "source": "wikisource",
        "id": "On_Physical_Lines_of_Force",
        "wikisource_type": "page",
        "author": "James Clerk Maxwell",
        "year": 1861,
        "title": "On Physical Lines of Force",
    },
    {
        "filename": "lorentz_em_theory.txt",
        "source": "gutenberg",
        "id": 70453,
        "author": "H.A. Lorentz",
        "year": 1891,
        "title": "Clerk Maxwell's Electromagnetic Theory",
    },
    {
        "filename": "lorentz_simplified_theory.txt",
        "source": "wikisource",
        "id": "Simplified_Theory_of_Electrical_and_Optical_Phenomena_in_Moving_Systems",
        "wikisource_type": "page",
        "author": "H.A. Lorentz",
        "year": 1899,
        "title": "Simplified Theory of Electrical and Optical Phenomena in Moving Systems",
    },
    {
        "filename": "poynting_transfer_energy_em.txt",
        "source": "wikisource",
        "id": "On_the_Transfer_of_Energy_in_the_Electromagnetic_Field",
        "wikisource_type": "page",
        "author": "John Henry Poynting",
        "year": 1884,
        "title": "On the Transfer of Energy in the Electromagnetic Field",
    },
    {
        "filename": "michelson_morley.txt",
        "source": "wikisource",
        "id": "On_the_Relative_Motion_of_the_Earth_and_the_Luminiferous_Ether",
        "wikisource_type": "page",
        "author": "Michelson & Morley",
        "year": 1887,
        "title": "On the Relative Motion of the Earth and the Luminiferous Ether",
    },
    {
        "filename": "fizeau_velocity_of_light.txt",
        "source": "wikisource",
        "id": "The_Hypotheses_Relating_to_the_Luminous_Aether",
        "wikisource_type": "page",
        "author": "Hippolyte Fizeau",
        "year": 1851,
        "title": "The Hypotheses Relating to the Luminous Aether (Fizeau)",
    },
    {
        "filename": "thomson_cathode_rays.txt",
        "source": "wikisource",
        "id": "Cathode_Rays",
        "wikisource_type": "page",
        "author": "J.J. Thomson",
        "year": 1897,
        "title": "Cathode Rays",
    },
    {
        "filename": "thomson_notes_recent_researches.txt",
        "source": "ia",
        "id": "notesonrecentres00thom",
        "author": "J.J. Thomson",
        "year": 1893,
        "title": "Notes on Recent Researches in Electricity and Magnetism",
    },
    {
        "filename": "heaviside_moving_charge_electric_magnetic.txt",
        "source": "wikisource",
        "id": "On_the_Electric_and_Magnetic_Effects_produced_by_the_Motion_of_Electrified_Bodies",
        "wikisource_type": "page",
        "author": "Oliver Heaviside",
        "year": 1881,
        "title": "On the Electric and Magnetic Effects produced by the Motion of Electrified Bodies",
    },
    {
        "filename": "heaviside_em_effects_moving_charge.txt",
        "source": "wikisource",
        "id": "Electromagnetic_effects_of_a_moving_charge",
        "wikisource_type": "page",
        "author": "Oliver Heaviside",
        "year": 1888,
        "title": "The Electromagnetic effects of a moving charge",
    },
    {
        "filename": "thomson_electrification_dielectric.txt",
        "source": "wikisource",
        "id": "Motion_of_Electrification_through_a_Dielectric",
        "wikisource_type": "page",
        "author": "J.J. Thomson",
        "year": 1889,
        "title": "On the Motion of Electrification through a Dielectric",
    },
    {
        "filename": "fitzgerald_electrified_ellipsoid.txt",
        "source": "wikisource",
        "id": "On_the_Steady_Motion_of_an_Electrified_Ellipsoid",
        "wikisource_type": "page",
        "author": "George FitzGerald",
        "year": 1897,
        "title": "On the Steady Motion of an Electrified Ellipsoid",
    },
    {
        "filename": "larmor_em_theory_moving_charges.txt",
        "source": "wikisource",
        "id": "Notes_on_the_Electro-magnetic_Theory_of_Moving_Charges",
        "wikisource_type": "page",
        "author": "W.B. Morton (on Larmor's work)",
        "year": 1896,
        "title": "Notes on the Electro-magnetic Theory of Moving Charges",
    },
    {
        "filename": "larmor_dynamical_theory_iii.txt",
        "source": "wikisource",
        "id": "Dynamical_Theory_of_the_Electric_and_Luminiferous_Medium_III",
        "wikisource_type": "page",
        "author": "Joseph Larmor",
        "year": 1897,
        "title": "A Dynamical Theory of the Electric and Luminiferous Medium, Part III",
    },
    {
        "filename": "tyndall_on_radiation.txt",
        "source": "wikisource",
        "id": "On_Radiation",
        "wikisource_type": "page",
        "author": "John Tyndall",
        "year": 1865,
        "title": "On Radiation (Rede Lecture)",
    },
    {
        "filename": "stokes_luminiferous_aether.txt",
        "source": "wikisource",
        "id": "On_the_Constitution_of_the_Luminiferous_Æther",
        "wikisource_type": "page",
        "author": "George Gabriel Stokes",
        "year": 1848,
        "title": "On the Constitution of the Luminiferous Æther",
    },
    # stokes_luminiferous_aether_ii: same Wikisource page as stokes_luminiferous_aether, skipped
    {
        "filename": "stokes_aberration_of_light.txt",
        "source": "wikisource",
        "id": "On_the_Aberration_of_Light",
        "wikisource_type": "page",
        "author": "George Gabriel Stokes",
        "year": 1845,
        "title": "On the Aberration of Light",
    },
    {
        "filename": "stokes_fresnels_theory_aberration.txt",
        "source": "wikisource",
        "id": "On_Fresnel's_Theory_of_the_Aberration_of_Light",
        "wikisource_type": "page",
        "author": "George Gabriel Stokes",
        "year": 1846,
        "title": "On Fresnel's Theory of the Aberration of Light",
    },
    {
        "filename": "scientific_memoirs_vol1.txt",
        "source": "wikisource",
        "id": "Scientific_Memoirs/1",
        "wikisource_type": "subpages",
        "author": "Various",
        "year": 1837,
        "title": "Scientific Memoirs, Vol 1",
    },
    {
        "filename": "scientific_memoirs_vol2.txt",
        "source": "wikisource",
        "id": "Scientific_Memoirs/2",
        "wikisource_type": "subpages",
        "author": "Various",
        "year": 1841,
        "title": "Scientific Memoirs, Vol 2",
    },
    {
        "filename": "scientific_memoirs_vol3.txt",
        "source": "wikisource",
        "id": "Scientific_Memoirs/3",
        "wikisource_type": "subpages",
        "author": "Various",
        "year": 1853,
        "title": "Scientific Memoirs, Vol 3",
    },
    {
        "filename": "somerville_mechanisms_heavens.txt",
        "source": "wikisource",
        "id": "A_Preliminary_Dissertation_on_the_Mechanisms_of_the_Heavens",
        "wikisource_type": "page",
        "author": "Mary Somerville",
        "year": 1831,
        "title": "A Preliminary Dissertation on the Mechanisms of the Heavens",
    },
    {
        "filename": "airy_elementary_treatise_optics.txt",
        "source": "wikisource",
        "id": "An_Elementary_Treatise_on_Optics",
        "wikisource_type": "subpages",
        "author": "George Biddell Airy",
        "year": 1831,
        "title": "An Elementary Treatise on Optics",
    },
    {
        "filename": "kimball_textbook_physics.txt",
        "source": "wikisource",
        "id": "Elementary_Text-book_of_Physics",
        "wikisource_type": "subpages",
        "author": "A.L. Kimball",
        "year": 1897,
        "title": "Elementary Text-book of Physics",
    },
    {
        "filename": "berry_short_history_astronomy.txt",
        "source": "wikisource",
        "id": "A_Short_History_of_Astronomy_(1898)",
        "wikisource_type": "subpages",
        "author": "Arthur Berry",
        "year": 1898,
        "title": "A Short History of Astronomy",
    },
    {
        "filename": "airy_popular_astronomy.txt",
        "source": "wikisource",
        "id": "Popular_Astronomy:_A_Series_of_Lectures_Delivered_at_Ipswich",
        "wikisource_type": "subpages",
        "author": "George Biddell Airy",
        "year": 1881,
        "title": "Popular Astronomy (Ipswich lectures)",
    },
    {
        "filename": "tyndall_fragments_of_science.txt",
        "source": "gutenberg",
        "id": 24527,
        "author": "John Tyndall",
        "year": 1871,
        "title": "Fragments of Science",
    },
    {
        "filename": "helmholtz_popular_lectures.txt",
        "source": "gutenberg",
        "id": 77725,
        "author": "Hermann von Helmholtz",
        "year": 1873,
        "title": "Popular Lectures on Scientific Subjects",
    },
    {
        "filename": "kelvin_molecular_tactics_crystal.txt",
        "source": "gutenberg",
        "id": 54976,
        "author": "Lord Kelvin",
        "year": 1894,
        "title": "The Molecular Tactics of a Crystal",
    },
    {
        "filename": "tyndall_contributions_molecular_physics.txt",
        "source": "ia",
        "id": "contributionsto00tyndgoog",
        "author": "John Tyndall",
        "year": 1872,
        "title": "Contributions to Molecular Physics in the Domain of Radiant Heat",
    },
    {
        "filename": "meyer_kinetic_theory_gases.txt",
        "source": "ia",
        "id": "kinetictheoryofg00meyeuoft",
        "author": "Oskar Emil Meyer",
        "year": 1899,
        "title": "The Kinetic Theory of Gases",
    },
    {
        "filename": "young_lectures_natural_philosophy.txt",
        "source": "ia",
        "id": "courseoflectures01younrich",
        "author": "Thomas Young",
        "year": 1845,
        "title": "A Course of Lectures on Natural Philosophy",
    },
    # ======================================================================
    # TIER 3: Shorter Works & Papers
    # ======================================================================
    {
        "filename": "clausius_memoir_motive_power.txt",
        "source": "wikisource",
        "id": "Scientific_Memoirs/1/Memoir_on_the_Motive_Power_of_Heat",
        "wikisource_type": "page",
        "author": "Rudolf Clausius",
        "year": 1850,
        "title": "Memoir on the Motive Power of Heat",
    },
    {
        "filename": "hypotheses_luminous_ether.txt",
        "source": "wikisource",
        "id": "The_Hypotheses_Relating_to_the_Luminous_Aether",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1899,
        "title": "The Hypotheses Relating to the Luminous Aether",
    },
    {
        "filename": "fitzgerald_ether_earths_atmosphere.txt",
        "source": "wikisource",
        "id": "The_Ether_and_the_Earth's_Atmosphere",
        "wikisource_type": "page",
        "author": "George FitzGerald",
        "year": 1889,
        "title": "The Ether and the Earth's Atmosphere",
    },
    {
        "filename": "bose_electric_waves.txt",
        "source": "wikisource",
        "id": "On_a_Complete_Apparatus_for_the_Study_of_the_Properties_of_Electric_Waves",
        "wikisource_type": "page",
        "author": "Jagadish Chandra Bose",
        "year": 1897,
        "title": "On a Complete Apparatus for Electric Waves",
    },
    {
        "filename": "lodge_dissipation_hertz_resonator.txt",
        "source": "wikisource",
        "id": "On_the_Dissipation_of_the_Electrical_Energy_of_the_Hertz_Resonator",
        "wikisource_type": "page",
        "author": "Oliver Lodge",
        "year": 1892,
        "title": "On the Dissipation of Electrical Energy of the Hertz Resonator",
    },
    {
        "filename": "comparative_review_gravitation.txt",
        "source": "wikisource",
        "id": "Comparative_Review_of_some_Dynamical_Theories_of_Gravitation",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1895,
        "title": "A Comparative Review of Dynamical Theories of Gravitation",
    },
    {
        "filename": "kinetic_theories_gravitation.txt",
        "source": "wikisource",
        "id": "Kinetic_Theories_of_Gravitation",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1895,
        "title": "Kinetic Theories of Gravitation",
    },
    {
        "filename": "preston_physical_theories_gravitation.txt",
        "source": "wikisource",
        "id": "Physical_Theories_of_Gravitation",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1895,
        "title": "Physical Theories of Gravitation",
    },
    {
        "filename": "le_sage_theory_gravitation.txt",
        "source": "wikisource",
        "id": "The_Le_Sage_Theory_of_Gravitation",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1895,
        "title": "The Le Sage Theory of Gravitation",
    },
    {
        "filename": "le_sage_theory_gravitation_2.txt",
        "source": "wikisource",
        "id": "Le_Sage's_Theory_of_Gravitation",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1895,
        "title": "Le Sage's Theory of Gravitation",
    },
    {
        "filename": "preston_vortex_atoms.txt",
        "source": "wikisource",
        "id": "Mutual_Action_of_Vortex_Atoms_and_Ultramundane_Corpuscles",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1879,
        "title": "On the Mutual Action of Vortex Atoms and Ultramundane Corpuscles",
    },
    {
        "filename": "browne_action_at_distance.txt",
        "source": "wikisource",
        "id": "On_Action_at_a_Distance_(Browne)",
        "wikisource_type": "page",
        "author": "J.H. Browne",
        "year": 1881,
        "title": "On Action at a Distance (Browne)",
    },
    {
        "filename": "preston_action_at_distance.txt",
        "source": "wikisource",
        "id": "On_Action_at_a_Distance_(Preston)",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1881,
        "title": "On Action at a Distance (Preston)",
    },
    {
        "filename": "preston_dynamical_conditions.txt",
        "source": "wikisource",
        "id": "On_some_Dynamical_Conditions",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1877,
        "title": "On some Dynamical Conditions of the Kinetic Theory of Gravitation",
    },
    {
        "filename": "preston_importance_experiments.txt",
        "source": "wikisource",
        "id": "On_the_Importance_of_Experiments",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1881,
        "title": "On the Importance of Experiments on Le Sage's Theory",
    },
    {
        "filename": "preston_crystallization.txt",
        "source": "wikisource",
        "id": "A_Suggestion_in_regard_to_Crystallization",
        "wikisource_type": "page",
        "author": "S. Tolver Preston",
        "year": 1880,
        "title": "A Suggestion in regard to Crystallization",
    },
    {
        "filename": "clifford_space_theory_matter.txt",
        "source": "wikisource",
        "id": "On_the_Space-Theory_of_Matter",
        "wikisource_type": "page",
        "author": "William Kingdon Clifford",
        "year": 1876,
        "title": "On the Space-Theory of Matter",
    },
    {
        "filename": "euler_space_and_time.txt",
        "source": "wikisource",
        "id": "Of_Space_and_Time",
        "wikisource_type": "page",
        "author": "Leonhard Euler",
        "year": 1763,
        "title": "Of Space and Time",
    },
    {
        "filename": "reid_essay_on_quantity.txt",
        "source": "wikisource",
        "id": "An_Essay_on_Quantity",
        "wikisource_type": "page",
        "author": "Thomas Reid",
        "year": 1748,
        "title": "An Essay on Quantity",
    },
    {
        "filename": "young_sound_and_light.txt",
        "source": "wikisource",
        "id": "Outlines_of_Experiments_and_Inquiries_Respecting_Sound_and_Light",
        "wikisource_type": "page",
        "author": "Thomas Young",
        "year": 1800,
        "title": "Outlines of Experiments Respecting Sound and Light",
    },
    {
        "filename": "maury_winds_currents.txt",
        "source": "wikisource",
        "id": "An_essay_on_the_winds_and_the_currents_of_the_ocean",
        "wikisource_type": "page",
        "author": "Matthew Fontaine Maury",
        "year": 1856,
        "title": "An essay on the winds and currents of the ocean",
    },
    {
        "filename": "foucault_earth_rotation.txt",
        "source": "wikisource",
        "id": "Note_relating_to_M._Foucault's_new_mechanical_proof_of_the_Rotation_of_the_Earth",
        "wikisource_type": "page",
        "author": "Various",
        "year": 1851,
        "title": "Note on Foucault's proof of Earth rotation",
    },
    {
        "filename": "abbott_flatland.txt",
        "source": "wikisource",
        "id": "Flatland_(first_edition)",
        "wikisource_type": "subpages",
        "author": "Edwin Abbott",
        "year": 1884,
        "title": "Flatland",
    },
    {
        "filename": "masson_atomic_theory_lucretius.txt",
        "source": "wikisource",
        "id": "The_Atomic_Theory_of_Lucretius",
        "wikisource_type": "subpages",
        "author": "John Masson",
        "year": 1884,
        "title": "The Atomic Theory of Lucretius",
    },
    {
        "filename": "voltaire_english_nation.txt",
        "source": "wikisource",
        "id": "Letters_concerning_the_English_Nation",
        "wikisource_type": "subpages",
        "author": "Voltaire",
        "year": 1733,
        "title": "Letters concerning the English Nation",
    },
    {
        "filename": "de_la_hire_treatise_on_eye.txt",
        "source": "wikisource",
        "id": "A_Treatise_on_the_Eye",
        "wikisource_type": "page",
        "author": "Philippe de La Hire",
        "year": 1759,
        "title": "A Treatise on the Eye",
    },
]


# ---------------------------------------------------------------------------
# HTML → plaintext helpers (no external dependencies)
# ---------------------------------------------------------------------------

def html_to_plaintext(raw_html: str) -> str:
    """Convert HTML to plaintext using regex. No external deps needed."""
    text = raw_html
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Convert <br>, <p>, <div>, <li>, heading tags to newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|tr|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<(p|div|li|h[1-6]|tr|blockquote)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Normalize whitespace: collapse runs of spaces/tabs (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg boilerplate header and footer."""
    # Find start marker
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip past the marker line
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    # Find end marker
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def _subpage_sort_key(title: str) -> tuple:
    """Sort subpages by extracting numeric/roman components from the last path segment."""
    parts = title.rsplit("/", 1)
    segment = parts[-1] if len(parts) > 1 else title

    # Try direct integer
    try:
        return (0, int(segment), "")
    except ValueError:
        pass

    # Try roman numerals at start (e.g., "Chapter_IV")
    roman_map = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
        "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
        "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
        "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
        "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30,
    }
    upper_seg = segment.upper().strip()
    if upper_seg in roman_map:
        return (0, roman_map[upper_seg], "")

    # Try to extract leading numbers
    m = re.match(r"(\d+)", segment)
    if m:
        return (0, int(m.group(1)), segment)

    # Fallback: alphabetical
    return (1, 0, segment)


# ---------------------------------------------------------------------------
# Downloader classes
# ---------------------------------------------------------------------------

class WikisourceDownloader:
    """Download works from Wikisource via the MediaWiki Parse API."""

    API_URL = "https://en.wikisource.org/w/api.php"
    RATE_LIMIT = 1.0  # seconds between requests

    def __init__(self, session: requests.Session):
        self.session = session
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self._last_request = time.time()

    def get_page_text(self, page_title: str) -> str | None:
        """Fetch a single Wikisource page and return plaintext."""
        self._rate_limit()
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
            "disablelimitreport": "true",
        }
        try:
            resp = self.session.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"  Failed to fetch Wikisource page '{page_title}': {e}")
            return None

        if "error" in data:
            log.warning(f"  Wikisource API error for '{page_title}': {data['error'].get('info', '')}")
            return None

        raw_html = data.get("parse", {}).get("text", {}).get("*", "")
        if not raw_html:
            return None

        return html_to_plaintext(raw_html)

    def get_subpages(self, prefix: str) -> list[str]:
        """List all subpages under a given prefix."""
        self._rate_limit()
        subpages = []
        params = {
            "action": "query",
            "list": "allpages",
            "apprefix": prefix + "/",
            "apnamespace": 0,
            "aplimit": 500,
            "format": "json",
        }
        try:
            resp = self.session.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("query", {}).get("allpages", []):
                title = page["title"]
                # Skip deeply nested subpages (e.g., page/chapter/section)
                # We want direct children of our prefix
                subpages.append(title)
        except Exception as e:
            log.warning(f"  Failed to list subpages for '{prefix}': {e}")

        # Sort subpages
        subpages.sort(key=_subpage_sort_key)
        return subpages

    def download_work(self, work: dict) -> str | None:
        """Download a complete work. Returns plaintext or None."""
        ws_type = work.get("wikisource_type", "page")
        page_id = work["id"]

        if ws_type == "page":
            log.info(f"  Fetching single page: {page_id}")
            return self.get_page_text(page_id)

        elif ws_type == "subpages":
            log.info(f"  Listing subpages for: {page_id}")
            subpages = self.get_subpages(page_id)

            if not subpages:
                # Try fetching the main page itself (it might have content)
                log.info(f"  No subpages found, trying main page: {page_id}")
                main_text = self.get_page_text(page_id)
                if main_text and len(main_text) > 500:
                    return main_text
                return None

            log.info(f"  Found {len(subpages)} subpages")
            parts = []
            for i, sp in enumerate(subpages):
                log.info(f"  Fetching subpage {i+1}/{len(subpages)}: {sp}")
                text = self.get_page_text(sp)
                if text and len(text.strip()) > 50:
                    parts.append(text)

            if not parts:
                return None
            return "\n\n".join(parts)

        return None


class GutenbergDownloader:
    """Download works from Project Gutenberg."""

    def __init__(self, session: requests.Session):
        self.session = session

    def download_work(self, work: dict) -> str | None:
        """Download a book by Gutenberg ID. Returns plaintext or None."""
        book_id = work["id"]
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        log.info(f"  Downloading from Gutenberg: {url}")

        try:
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            text = resp.text
        except Exception as e:
            log.warning(f"  Failed to download Gutenberg book {book_id}: {e}")
            # Try alternate URL format
            alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            log.info(f"  Trying alternate URL: {alt_url}")
            try:
                resp = self.session.get(alt_url, timeout=60)
                resp.raise_for_status()
                text = resp.text
            except Exception as e2:
                log.warning(f"  Alternate URL also failed: {e2}")
                return None

        return strip_gutenberg_header_footer(text)


class InternetArchiveDownloader:
    """Download OCR text from Internet Archive."""

    def __init__(self, session: requests.Session):
        self.session = session

    def download_work(self, work: dict) -> str | None:
        """Download _djvu.txt for an IA identifier. Returns text or None."""
        identifier = work["id"]
        url = f"https://archive.org/download/{identifier}/{identifier}_djvu.txt"
        log.info(f"  Downloading from Internet Archive: {url}")

        try:
            resp = self.session.get(url, timeout=120)
            resp.raise_for_status()
            return resp.text.strip()
        except Exception as e:
            log.warning(f"  Failed to download IA item '{identifier}': {e}")
            # Try to find the djvu.txt file in the file listing
            log.info(f"  Trying to locate text file in IA metadata...")
            try:
                meta_url = f"https://archive.org/metadata/{identifier}/files"
                resp = self.session.get(meta_url, timeout=30)
                resp.raise_for_status()
                files = resp.json().get("result", [])
                txt_files = [f["name"] for f in files if f["name"].endswith("_djvu.txt")]
                if txt_files:
                    alt_name = txt_files[0]
                    alt_url = f"https://archive.org/download/{identifier}/{alt_name}"
                    log.info(f"  Found alternate text file: {alt_name}")
                    resp = self.session.get(alt_url, timeout=120)
                    resp.raise_for_status()
                    return resp.text.strip()
            except Exception as e2:
                log.warning(f"  Metadata lookup also failed: {e2}")
            return None


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect canonical pre-1900 physics texts from Wikisource, Gutenberg, and Internet Archive."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/physics_books",
        help="Output directory for downloaded texts (default: data/physics_books)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all works without downloading",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        help="Only download specific files (by filename)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["wikisource", "gutenberg", "ia"],
        help="Only download from a specific source",
    )
    args = parser.parse_args()

    if args.no_skip_existing:
        args.skip_existing = False

    output_dir = Path(args.output_dir)

    # Filter works
    works = CURATED_WORKS
    if args.only:
        works = [w for w in works if w["filename"] in args.only]
        if not works:
            log.error(f"No works matched --only {args.only}")
            return
    if args.source:
        works = [w for w in works if w["source"] == args.source]

    # Dry run: just list
    if args.dry_run:
        print(f"\n{'#':>3}  {'Source':<12} {'Filename':<55} {'Author':<25} {'Year':<6} Title")
        print("-" * 140)
        for i, w in enumerate(works, 1):
            existing = "EXISTS" if (output_dir / w["filename"]).exists() else ""
            print(
                f"{i:>3}  {w['source']:<12} {w['filename']:<55} "
                f"{w['author']:<25} {str(w['year']):<6} {w['title']}  {existing}"
            )
        print(f"\nTotal: {len(works)} works")

        # Count by source
        from collections import Counter
        source_counts = Counter(w["source"] for w in works)
        for src, count in sorted(source_counts.items()):
            print(f"  {src}: {count}")
        return

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Pre1900PhysicsCollector/1.0 (academic research; michael@example.com)"
    })

    # Initialize downloaders
    downloaders = {
        "wikisource": WikisourceDownloader(session),
        "gutenberg": GutenbergDownloader(session),
        "ia": InternetArchiveDownloader(session),
    }

    # Download
    success = 0
    skipped = 0
    failed = 0

    for i, work in enumerate(works, 1):
        filename = work["filename"]
        filepath = output_dir / filename
        source = work["source"]

        log.info(f"[{i}/{len(works)}] {work['title']} ({work['author']}, {work['year']})")

        # Skip existing
        if args.skip_existing and filepath.exists() and filepath.stat().st_size > 0:
            log.info(f"  Skipping (already exists): {filepath}")
            skipped += 1
            continue

        downloader = downloaders[source]
        text = downloader.download_work(work)

        if text and len(text.strip()) > 100:
            filepath.write_text(text, encoding="utf-8")
            size_kb = filepath.stat().st_size / 1024
            log.info(f"  Saved: {filepath} ({size_kb:.1f} KB)")
            success += 1
        else:
            log.warning(f"  FAILED or empty: {filename}")
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")
    print(f"  Total:    {len(works)}")
    print(f"  Output:   {output_dir.resolve()}")


if __name__ == "__main__":
    main()
