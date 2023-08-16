#!/bin/bash
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
systemctl start elk_model.service
systemctl restart cron.service
tail -f /dev/null
