function mail2me(receiver,subject,content)
    smtp_server='smtp.gmail.com';
    smtp_port='465';
    smtp_username='...';
    smtp_password='...';
    setpref('Internet','SMTP_Server',smtp_server);
    setpref('Internet','SMTP_Username',smtp_username);
    setpref('Internet','SMTP_Password',smtp_password);
    setpref('Internet','E_mail',smtp_username);
    props = java.lang.System.getProperties;
    props.setProperty('mail.smtp.auth','true');
    props.setProperty('mail.smtp.socketFactory.class', ...
        'javax.net.ssl.SSLSocketFactory');
    props.setProperty('mail.smtp.socketFactory.port',smtp_port);
    sendmail(receiver,subject,content);
end