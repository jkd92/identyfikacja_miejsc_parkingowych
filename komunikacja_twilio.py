from twilio.rest import Client

def send_bulk_sms(account_sid, auth_token, msg_service_sid, body, numbers):
    # Inicjalizacja klienta Twilio
    client = Client(account_sid, auth_token)
    
    # Iteracja przez listę numerów telefonów
    for number in numbers:
        # Wysyłanie wiadomości SMS
        message = client.messages.create(
            body=body,  # Treść wiadomości
            messaging_service_sid=msg_service_sid,  # ID usługi wiadomości (opcjonalne)
            to=number  # Numer docelowy
        )
        
        print(f"Wiadomość wysłana do {number}, SID {message.sid}")

if __name__ == "__main__":
    # Twoje ID konta Twilio i token autoryzacyjny
    account_sid = "TWILIO_ACCOUNT_SID_HERE"
    auth_token = "TWILIO_AUTH_TOKEN_HERE"
    
    # Treść wiadomości SMS
    message_text = "Dzień dobry, to jest testowa wiadomość."
    
    # Lista numerów telefonów
    phone_numbers_list = ["+1111111", "+33333333"]
    
    # Wysyłanie wiadomości SMS do wszystkich numerów z listy
    send_bulk_sms(account_sid, auth_token, msg_service_sid, message_text, phone_numbers_list)
