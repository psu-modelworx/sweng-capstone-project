from celery import shared_task

from django.core.mail import EmailMessage
from django.conf import settings

from .models import EmailTask

import logging
logger = logging.getLogger(__name__)

@shared_task(bind=True)
def send_account_creation_email_task(self, useremail, username):
    logger.info("Task: Send email to admins for account creation request.")

    ## Update Tasks
    task_record = EmailTask.objects.filter(task_id=self.request.id).first()
    if task_record:
        task_record.status = "STARTED"
        task_record.save()

    msg_body = """
    The following user has requested Modelworx access:
    User Email Address:  {0}
    Username:  {1}
    """.format(useremail, username)

    email_admins = settings.EMAIL_ADMINS.split(',')
    print("EMAIL ADMINS: {0}".format(email_admins))
    email_sender = settings.EMAIL_SENDER

    message = EmailMessage(
      'Modelworx Account Request',
      msg_body,
      email_sender,
      email_admins
    )
    try:
      message.send()
    except Exception as e:
        msg = 'Exception sending emails to admin: {0}'.format(e)
        logger.error(msg)
        if task_record:
            task_record.status = "FAILURE"
            task_record.result_message = msg
            task_record.save()
        return {"message": msg, "status":500}
    
    msg = "Email admins task successfully completed!"
    if task_record:
        task_record.status = "SUCCESS"
        task_record.result_message = msg
        task_record.save()

    return {"message": msg, "status": 200}