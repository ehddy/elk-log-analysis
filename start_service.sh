#!/bin/bash
systemctl start elk_model.service
systemctl restart cron.service
tail -f /dev/null
