import { Resend } from "resend";

function getResend() {
  return new Resend(process.env.RESEND_API_KEY || "re_dummy");
}

const APP_NAME = "Ucotron";
const APP_URL = process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";
const FROM_EMAIL = process.env.RESEND_FROM_EMAIL || "noreply@ucotron.ai";

interface SendEmailParams {
  to: string;
  subject: string;
  html: string;
}

export async function sendEmail({ to, subject, html }: SendEmailParams) {
  if (!process.env.RESEND_API_KEY) {
    console.warn("RESEND_API_KEY not set - email would be sent to:", to);
    return { success: true, mock: true };
  }

  try {
    const { data, error } = await getResend().emails.send({
      from: FROM_EMAIL,
      to,
      subject,
      html,
    });

    if (error) {
      console.error("Failed to send email:", error);
      return { success: false, error };
    }

    return { success: true, data };
  } catch (err) {
    console.error("Failed to send email:", err);
    return { success: false, error: err };
  }
}

export function sendVerificationEmail(params: {
  user: { email: string; name: string };
  url: string;
  token: string;
}) {
  const { user, url } = params;
  return sendEmail({
    to: user.email,
    subject: `Verify your email for ${APP_NAME}`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .container { background: #0a0a0a; border-radius: 12px; padding: 40px; border: 1px solid #262626; }
            .logo { font-size: 24px; font-weight: bold; color: #a855f7; margin-bottom: 24px; letter-spacing: 0.1em; text-transform: uppercase; }
            .title { color: #fff; font-size: 24px; margin-bottom: 16px; }
            .text { color: #a1a1aa; margin-bottom: 24px; }
            .button { display: inline-block; background: #a855f7; color: #fff; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }
            .footer { color: #71717a; font-size: 12px; margin-top: 24px; border-top: 1px solid #262626; padding-top: 16px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="logo">${APP_NAME}</div>
            <h1 class="title">Verify your email</h1>
            <p class="text">Hi ${user.name},</p>
            <p class="text">Thanks for signing up! Please click the button below to verify your email address.</p>
            <a href="${url}" class="button">Verify Email</a>
            <p class="text" style="margin-top: 24px;">Or copy this link to your browser:</p>
            <p class="text" style="word-break: break-all; color: #a855f7;">${url}</p>
            <p class="footer">If you didn't create an account with ${APP_NAME}, you can safely ignore this email.</p>
          </div>
        </body>
      </html>
    `,
  });
}

export function sendPasswordResetEmail(params: {
  user: { email: string; name: string };
  url: string;
  token: string;
}) {
  const { user, url } = params;
  return sendEmail({
    to: user.email,
    subject: `Reset your ${APP_NAME} password`,
    html: `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }
            .container { background: #0a0a0a; border-radius: 12px; padding: 40px; border: 1px solid #262626; }
            .logo { font-size: 24px; font-weight: bold; color: #a855f7; margin-bottom: 24px; letter-spacing: 0.1em; text-transform: uppercase; }
            .title { color: #fff; font-size: 24px; margin-bottom: 16px; }
            .text { color: #a1a1aa; margin-bottom: 24px; }
            .button { display: inline-block; background: #a855f7; color: #fff; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }
            .footer { color: #71717a; font-size: 12px; margin-top: 24px; border-top: 1px solid #262626; padding-top: 16px; }
            .warning { color: #f97316; font-size: 12px; }
          </style>
        </head>
        <body>
          <div class="container">
            <div class="logo">${APP_NAME}</div>
            <h1 class="title">Reset your password</h1>
            <p class="text">Hi ${user.name},</p>
            <p class="text">We received a request to reset your password. Click the button below to create a new one.</p>
            <a href="${url}" class="button">Reset Password</a>
            <p class="text" style="margin-top: 24px;">Or copy this link to your browser:</p>
            <p class="text" style="word-break: break-all; color: #a855f7;">${url}</p>
            <p class="warning">This link will expire in 1 hour.</p>
            <p class="footer">If you didn't request a password reset, you can safely ignore this email.</p>
          </div>
        </body>
      </html>
    `,
  });
}
