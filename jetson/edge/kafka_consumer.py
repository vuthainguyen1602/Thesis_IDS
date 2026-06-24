#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge.role_pipelines import create_pipeline


if __name__ == "__main__":
    create_pipeline().run()
