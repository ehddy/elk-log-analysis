#!/bin/bash
systemctl start elk_web.service
systemctl restart cron.service
