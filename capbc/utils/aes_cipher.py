import base64

from Crypto.Cipher import AES

__all__ = ["AESCipher"]


class AESCipher(object):
    """AES encrypt and decrypt.

    Parameters
    ----------
    key (str): Encryption key, AES key must be either 16, 24, or 32 bytes long
    """

    def __init__(self, key=b"ToBeNo.1\x08\x08\x08\x08\x08\x08\x08\x08"):
        self._key = key
        self._iv = key

    def encrypt(self, raw: str) -> str:
        """Encrypt str.

        Args:
            raw (str): raw text to be encrypted

        Returns:
            str: encryption base64 str
        """
        cipher = AES.new(self._key, AES.MODE_CBC, self._iv)
        encryption = cipher.encrypt(self._pad(raw).encode())
        return base64.b64encode(encryption).decode()

    def decrypt(self, enc: str) -> str:
        """Decrypt the encryption.

        Args:
            enc (str): encryption base64 str

        Returns:
            str: raw text to be encrypted
        """
        cipher = AES.new(self._key, AES.MODE_CBC, self._iv)
        base64_str = base64.b64decode(enc)
        text = cipher.decrypt(base64_str).decode()
        return self._unpad(text)

    def _pad(self, text):
        text_length = len(text)
        amount_to_pad = AES.block_size - (text_length % AES.block_size)
        if amount_to_pad == 0:
            amount_to_pad = AES.block_size
        pad = chr(amount_to_pad)
        return text + pad * amount_to_pad

    def _unpad(self, text):
        pad = ord(text[-1])
        return text[:-pad]
