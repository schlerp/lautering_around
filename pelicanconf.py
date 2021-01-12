#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = "schlerp"
SITENAME = "This is how you get ants!"
SITESUBTITLE = ""
SITEURL = ""

GOOGLE_ANALYTICS = "G-QED612WFHT"

PATH = "content"

TIMEZONE = "Australia/Darwin"

DEFAULT_LANG = "en"

# DISQUS_SITENAME = 'schlerpblog'

DISPLAY_CATEGORIES_ON_MENU = False

TYPOGRIFY = True

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TAGS_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Pagination
DEFAULT_PAGINATION = 3

# theme settings
THEME = "./theme_medius"

# HTML metadata
SITEDESCRIPTION = "A blog about brewing beer, cooking, and assorted other things."

# all defaults to True.
DISPLAY_HEADER = True
DISPLAY_FOOTER = True
DISPLAY_HOME = True
DISPLAY_MENU = True

# provided as examples, they make 'clean' urls. used by MENU_INTERNAL_PAGES.
TAGS_URL = "tags"
TAGS_SAVE_AS = "tags/index.html"
AUTHORS_URL = "authors"
AUTHORS_SAVE_AS = "authors/index.html"
CATEGORIES_URL = "categories"
CATEGORIES_SAVE_AS = "categories/index.html"
ARCHIVES_URL = "archives"
ARCHIVES_SAVE_AS = "archives/index.html"

# use those if you want pelican standard pages to appear in your menu
MENU_INTERNAL_PAGES = (
    # ('Tags', TAGS_URL, TAGS_SAVE_AS),
    ("Authors", AUTHORS_URL, AUTHORS_SAVE_AS),
    ("Categories", CATEGORIES_URL, CATEGORIES_SAVE_AS),
    # ('Archives', ARCHIVES_URL, ARCHIVES_SAVE_AS),
)

# plugins
PLUGIN_PATHS = [
    "./pelican-plugins",
]
PLUGINS = [
    "readtime",
]

# theme specific
MEDIUS_CATEGORIES = {
    "Brewing": {
        "slug": "brewing",
        "description": "Brew days, recipes, and general thoughts about brewing beer",
        "logo": "/images/nzipa/hops.jpg",
        "thumbnail": "/images/nzipa/hops.jpg",
    },
    "Baking": {
        "slug": "baking",
        "description": "Baking bread and other pastries",
        "logo": "/images/crustycob/cob.jpg",
        "thumbnail": "/images/crustycob/cob.jpg",
    },
    "Cooking": {
        "slug": "cooking",
        "description": "Baking bread and other pastries",
        "logo": "/images/mushyrisotto/mushroom_stock.jpg",
        "thumbnail": "/images/mushyrisotto/mushroom_stock.jpg",
    },
    "Electronics": {
        "slug": "electronics",
        "description": "Electronics projects and ideas",
        "logo": "/images/fender5e3/amp_speaker.jpg",
        "thumbnail": "/images/fender5e3/amp_speaker.jpg",
    },
}

MEDIUS_AUTHORS = {
    "Schlerp": {
        "description": """
            I am patty, I brew beer and ferment stuff. I'm also into electronics and FOSS as a hobby. Professionally I am a python programmer & data scientist/architect.
        """,
        "cover": "/images/author/skinny_dipping1.jpg",
        "image": "/images/author/ash_patty_piwis_120x120.jpg",
        "links": (("github", "https://github.com/schlerp"),),
    },
    "Goonty": {
        "description": """
            I am Ashlee, I like to make muffins. Professionally I am a nurse!
        """,
        "cover": "/images/author/umbawarra.jpg",
        "image": "/images/author/ashlee.jpg",
        "links": (("instagram", "https://instagram.com/ashleelauren92/"),),
    },
}
